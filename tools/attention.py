# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test

from mmpose.datasets import build_dataloader, build_dataset
from distilpose.models import build_posenet

from einops import rearrange

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def print_model_structure(model, feature_extractor):
    """모델의 구조를 더 자세히 출력하는 함수"""
    def get_layers(model, prefix=''):
        for name, module in model.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            feature_extractor.log(f"\nLayer: {new_prefix}")
            feature_extractor.log(f"Type: {type(module).__name__}")
            feature_extractor.log(f"Parameters: {module}")
            if len(list(module.children())) > 0:  # 하위 모듈이 있는 경우
                get_layers(module, new_prefix)

    feature_extractor.log("\n=== Detailed Model Structure ===")
    get_layers(model)


class FeatureExtractor:
    def __init__(self):
        self.features = {}
        self.attentions = {}
        self.batch_count = 0
        self.max_batches = 20  # 20개의 배치를 처리하도록 수정
        self.hooks = []
        self.processed_layers = set()
    
    def save_attention(self, name):
        def hook(module, input, output):
            if self.batch_count < self.max_batches:  # 첫 20개 배치만 처리
                # 첫 두 개의 이미지만 선택
                output = output[0:2]  # 배치에서 2개의 이미지만 선택
                
                # Q, K, V를 분리하여 attention 계산
                qkv = output.chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=8), qkv)
                dots = torch.matmul(q, k.transpose(-1, -2))
                attn = F.softmax(dots, dim=-1)
                
                # 배치별로 attention 저장
                if name not in self.attentions:
                    self.attentions[name] = []
                self.attentions[name].append(attn.detach())
                print(f"\nSaved attention for {name}, shape: {attn.shape}")
                
                # 이 layer가 처리되었음을 표시
                self.processed_layers.add(name)
                
                # 모든 transformer layer가 처리되었는지 확인
                if len(self.processed_layers) == len(self.hooks):
                    self.processed_layers.clear()  # 다음 배치를 위해 초기화
                    self.batch_count += 1  # 배치 카운터 증가
                    print(f"Processed batch {self.batch_count}/{self.max_batches}")
                    
                    # 모든 배치 처리 완료 후 hook 제거
                    if self.batch_count >= self.max_batches:
                        for h in self.hooks:
                            h.remove()
                        self.hooks.clear()
                        print("Completed processing all batches")
        return hook


def visualize_attention_maps(attention, save_path, title, data=None, batch_idx=None):
    B, H, N, _ = attention.shape
    print(f"\nVisualizing attention map with shape: {attention.shape}")
    
    if data is not None and 'img' in data:
        # layer 이름 추출
        layer_name = title.split()[0]  # transformer_layerX 형태로 추출
        
        # batch별, layer별 폴더 생성
        base_save_dir = os.path.dirname(save_path)
        layer_save_dir = os.path.join(base_save_dir, f'batch_{batch_idx}', layer_name)
        os.makedirs(layer_save_dir, exist_ok=True)
        
        # 이미지 처리
        img = data['img'][0].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 이미지 크기
        img_height, img_width = img.shape[:2]
        
        # Feature map dimensions
        feature_h, feature_w = 32, 24  # 64/2, 48/2 (patch size가 2x2이므로)
        
        # attention map 계산 (17개의 keypoint token에 대해 각각)
        attn_mean = attention[0].mean(0)  # head 평균
        
        # keypoint attention 추출 [17, num_patches]
        num_keypoints = 17
        keypoint_attns = attn_mean[:num_keypoints, num_keypoints:]  # [17, 768]
        
        print(f"Processing {layer_name}")
        print(f"Attention shape after processing: {keypoint_attns.shape}")
        print(f"Feature map dimensions: {feature_h}x{feature_w}")
        
        # 모든 keypoint의 attention 합 계산 및 시각화
        all_keypoints_attn = keypoint_attns.mean(0).reshape(feature_h, feature_w)  # [32, 24]
        all_keypoints_attn_map = cv2.resize(all_keypoints_attn.cpu().numpy(),
                                          (img_width, img_height),
                                          interpolation=cv2.INTER_LINEAR)
        
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(all_keypoints_attn_map, alpha=0.5, cmap='jet')
        plt.title(f"All Keypoints Combined Attention")
        plt.colorbar()
        plt.axis('off')
        
        plt.savefig(os.path.join(layer_save_dir, "all_keypoints_attention.png"))
        plt.close()
        
        # 각 keypoint에 대한 attention map 시각화 (기존 코드)
        for kpt_idx in range(num_keypoints):
            patch_attn = keypoint_attns[kpt_idx].reshape(feature_h, feature_w)
            attn_map = cv2.resize(patch_attn.cpu().numpy(), 
                                (img_width, img_height), 
                                interpolation=cv2.INTER_LINEAR)
            
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(attn_map, alpha=0.5, cmap='jet')
            plt.title(f"Keypoint {kpt_idx} Attention")
            plt.colorbar()
            plt.axis('off')
            
            plt.savefig(os.path.join(layer_save_dir, f"keypoint_{kpt_idx}_attention.png"))
            plt.close()

        # 원본 이미지와 keypoint 시각화 (기존 코드)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if 'target' in data and 'target_weight' in data:
            target = data['target'][0].cpu().numpy()
            target_weight = data['target_weight'][0].cpu().numpy()
            scale_x = img_width / target.shape[2]
            scale_y = img_height / target.shape[1]
            for idx in range(target.shape[0]):
                if target_weight[idx] > 0:
                    heatmap = target[idx]
                    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y
                    plt.plot(x_scaled, y_scaled, 'ro', markersize=3)
        
        plt.title("Original Image with Keypoints")
        plt.axis('off')
        plt.savefig(os.path.join(layer_save_dir, "original_with_keypoints.png"))
        plt.close()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                               osp.splitext(osp.basename(args.config))[0])

    # attention map 저장 디렉토리 생성
    save_dir = os.path.join(cfg.work_dir, 'attention_maps', 'mct')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nAttention maps will be saved to: {save_dir}")
    print(f"Config work_dir: {cfg.work_dir}")
    print(f"Full save path example: {os.path.join(save_dir, 'transformer_layer0_batch0_keypoint1.png')}")
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 데이터셋 설정에서 augmentation 완전히 제거
    if cfg.data.test.get('pipeline', None):
        # 최소한의 필수 transform만 유지
        new_pipeline = []
        for transform in cfg.data.test.pipeline:
            # LoadImageFromFile: 이미지 로드
            # TopDownGetBboxCenterScale: bbox 정보 처리 (필수)
            # TopDownAffine: 기본 affine 변환 (필수)
            # ToTensor: numpy -> torch tensor 변환 (추가)
            # NormalizeTensor: 텐서 정규화
            # Collect: 결과 수집
            if transform['type'] in [
                'LoadImageFromFile',
                'TopDownGetBboxCenterScale',
                'TopDownAffine',
                'ToTensor',
                'NormalizeTensor',
                'Collect'
            ]:
                new_pipeline.append(transform)
        
        # ToTensor transform이 없다면 추가
        has_to_tensor = any(t['type'] == 'ToTensor' for t in new_pipeline)
        if not has_to_tensor:
            # NormalizeTensor 이전에 ToTensor 추가
            normalize_idx = next(i for i, t in enumerate(new_pipeline) 
                               if t['type'] == 'NormalizeTensor')
            new_pipeline.insert(normalize_idx, dict(type='ToTensor'))
            
        cfg.data.test.pipeline = new_pipeline
    
    # attention 데이터셋 사용
    if hasattr(cfg.data, 'attention'):
        dataset = build_dataset(cfg.data.attention)
    else:
        # attention pipeline이 없는 경우 기존 test pipeline 사용
        dataset = build_dataset(cfg.data.test)
    
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    data_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('train_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **data_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    
    # Feature Extractor 설정
    feature_extractor = FeatureExtractor()
    
    # transformer layer에 대해서만 hook 등록
    print("\n=== Registering hooks for transformer layers ===")
    
    for name, module in model.named_modules():
        if 'transformer_layer' in name and 'to_qkv' in name:
            try:
                hook = module.register_forward_hook(
                    feature_extractor.save_attention(name))
                feature_extractor.hooks.append(hook)
                print(f"Successfully registered hook for: {name}")
            except Exception as e:
                print(f"Error registering hook for {name}: {e}")
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = []
        prog_bar = mmcv.ProgressBar(len(data_loader))
        
        for batch_idx, data in enumerate(data_loader):
            with torch.no_grad():
                output = model(return_loss=False, **data)
                
                # "Processing batch 0" 메시지가 출력되는 시점에 이미지 저장
                if "Processing batch 0" in str(output):
                    print("\n=== 이미지 데이터 디버깅 ===")
                    try:
                        img = data['img'][0][0].cpu().numpy()
                        print(f"\n디버깅 정보:")
                        print(f"이미지 shape: {img.shape}")
                        print(f"이미지 dtype: {img.dtype}")
                        print(f"이미지 값 범위: [{img.min()}, {img.max()}]")
                        
                        # grayscale 이미지를 3채널로 변환
                        img = np.stack([img] * 3, axis=-1)  # (H, W) -> (H, W, 3)
                        
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = ((img * std) + mean) * 255
                        img = img.astype(np.uint8)
                        
                        original_path = os.path.join(save_dir, "original_image.png")
                        cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        print(f"원본 이미지 저장 완료: {original_path}")
                        
                    except Exception as e:
                        print(f"이미지 저장 중 오류 발생: {str(e)}")
                        print(f"상세 에러: {e.__class__.__name__}")
                        import traceback
                        print(traceback.format_exc())
                
                outputs.extend(output)
                prog_bar.update()
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        
        def process_batch_distributed(model, data, batch_idx):
            with torch.no_grad():
                output = model(return_loss=False, **data)
                
                if batch_idx >= 20:  # 20개 배치까지만 처리
                    return output
                
                if get_dist_info()[0] == 0 and batch_idx < 20:
                    print(f"\nProcessing batch {batch_idx}")
                    
                    # MCT attention maps 처리
                    for att_name, attention_list in feature_extractor.attentions.items():
                        if 'transformer_layer' in att_name:
                            # 현재 배치의 attention map 가져오기
                            attention = attention_list[batch_idx]
                            
                            # 각 이미지에 대해 개별적으로 처리
                            for img_idx in range(2):  # 배치의 첫 두 이미지만 처리
                                save_path = os.path.join(save_dir, f'{att_name}')
                                
                                # 이미지별 attention map 시각화
                                single_attention = attention[img_idx:img_idx+1]  # [1, H, N, N] 형태로 유지
                                
                                # data에서 해당 이미지 데이터만 추출
                                single_data = {
                                    'img': data['img'][img_idx:img_idx+1],
                                    'target': data['target'][img_idx:img_idx+1] if 'target' in data else None,
                                    'target_weight': data['target_weight'][img_idx:img_idx+1] if 'target_weight' in data else None
                                }
                                
                                visualize_attention_maps(single_attention, save_path, 
                                                      f"{att_name} (Batch {batch_idx}, Image {img_idx})", 
                                                      data=single_data,
                                                      batch_idx=f"{batch_idx}_img{img_idx}")
                
                return output
        
        # 데이터 처리
        outputs = []
        rank, _ = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(data_loader))
        
        for batch_idx, data in enumerate(data_loader):
            output = process_batch_distributed(model, data, batch_idx)
            outputs.extend(output)
            if rank == 0:
                prog_bar.update()

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
            
        print("\nAttention map visualization completed. Skipping evaluation.")
        
        # 기존 evaluation 코드 제거/주석 처리
        # eval_config = cfg.get('evaluation', {})
        # eval_config = merge_configs(eval_config, dict(metric=args.eval))
        # results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
        # for k, v in sorted(results.items()):
        #     print(f'{k}: {v}')


if __name__ == '__main__':
    main()
