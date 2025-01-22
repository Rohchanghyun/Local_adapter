# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import glob
import gc
import traceback

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
    def __init__(self, save_path=None):
        self.features = {}
        self.attentions = {}
        self.visual_tokens = {}
        self.hooks = []
        self.processed_layers = set()
        self.save_path = save_path
        self.current_batch = -1
        self.batch_count = 0  # batch_count를 인스턴스 변수로 이동
    
    def save_attention(self, name):
        def hook(module, input, output):
            qkv = output.chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=8), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2))
            attn = F.softmax(dots, dim=-1)
            
            # CPU로 즉시 이동하고 필요없는 텐서는 즉시 삭제
            if name not in self.attentions:
                self.attentions[name] = []
            self.attentions[name].append(attn.cpu().detach())
            
            # 중간 텐서들 명시적으로 삭제
            del qkv, q, k, v, dots
            torch.cuda.empty_cache()
            
            self.processed_layers.add(name)
            
            if len(self.processed_layers) == len(self.hooks):
                # 배치 처리가 끝나면 이전 데이터 삭제
                self.clear_previous_batch()
        return hook

    def save_visual_tokens(self, name):
        def hook(module, input, output):
            # self를 통해 인스턴스 변수에 접근
            if self.current_batch != self.batch_count:  # feature_extractor 대신 self 사용
                base_save_dir = os.path.dirname(self.save_path)
                tokens_dir = os.path.join(base_save_dir, 'mct', 'visual_tokens')
                os.makedirs(tokens_dir, exist_ok=True)
                
                visual_token = input[0][0].cpu().detach()
                tokens_path = os.path.join(tokens_dir, f"{self.batch_count:04d}.pt")
                torch.save(visual_token, tokens_path)
                
                if name not in self.visual_tokens:
                    self.visual_tokens[name] = []
                self.visual_tokens[name].append(visual_token)
                
                self.current_batch = self.batch_count
            
            torch.cuda.empty_cache()
        return hook

    def clear_previous_batch(self):
        """배치 처리 후 정리"""
        self.attentions.clear()
        self.visual_tokens.clear()
        self.processed_layers.clear()
        self.batch_count += 1  # 배치 카운터 증가
        torch.cuda.empty_cache()
        gc.collect()

    def remove_hooks(self):
        """모든 hook 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def visualize_attention_maps(attention, save_path, title, data=None, batch_idx=None, feature_extractor=None):
    B, H, N, _ = attention.shape
    #print(f"\nVisualizing attention map with shape: {attention.shape}")
    
    if data is not None and 'img' in data:
        # 기본 저장 디렉토리 설정
        base_save_dir = os.path.dirname(save_path)
        
        # 각각의 저장 디렉토리 생성
        images_dir = os.path.join(base_save_dir, 'images')
        attention_dir = os.path.join(base_save_dir, 'attention_maps')
        tokens_dir = os.path.join(base_save_dir, 'visual_tokens')
        
        for dir_path in [images_dir, attention_dir, tokens_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 현재 이미지의 번호 설정 (기존 파일 수를 확인하여 결정)
        existing_files = len(glob.glob(os.path.join(images_dir, '*.png')))
        current_idx = existing_files
        
        # 파일 이름 설정
        base_filename = f"{current_idx:04d}"  # 4자리 숫자로 패딩 (0000, 0001, ...)
        
        # 이미지 처리 및 저장
        img = data['img'][0].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 원본 이미지 저장
        img_path = os.path.join(images_dir, f"{base_filename}.png")
        cv2.imwrite(img_path, img)
        #print(f"Saved original image to: {img_path}")
        
        # Attention map 계산 및 저장
        attn_mean = attention[0].mean(0)
        num_keypoints = 17
        keypoint_attns = attn_mean[:num_keypoints, num_keypoints:]
        all_keypoints_attn = keypoint_attns.mean(0).reshape(32, 24)
        
        # Attention map 저장
        plt.figure(figsize=(5, 5))
        plt.imshow(all_keypoints_attn.cpu().numpy(), cmap='jet')
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        
        # dpi 설정으로 이미지 크기 조절
        height, width = all_keypoints_attn.shape
        dpi = plt.gcf().get_dpi()
        plt.gcf().set_size_inches(width/dpi, height/dpi)
        
        # attention map 저장
        attn_path = os.path.join(attention_dir, f"{base_filename}.png")
        plt.savefig(attn_path, 
                   bbox_inches='tight', 
                   pad_inches=0,
                   dpi=dpi)
        plt.close()
        #print(f"Saved attention map to: {attn_path}")
        
        # Visual tokens 저장 부분 수정 및 디버깅 추가
        if feature_extractor and hasattr(feature_extractor, 'visual_tokens'):
            #print("\nChecking visual tokens:")
            #print(f"Available tokens: {list(feature_extractor.visual_tokens.keys())}")
            
            for name, tokens_list in feature_extractor.visual_tokens.items():
                #print(f"Processing tokens for {name}")
                #print(f"Tokens list length: {len(tokens_list)}")
                
                if tokens_list and batch_idx < len(tokens_list):
                    tokens = tokens_list[batch_idx]
                    tokens_path = os.path.join(tokens_dir, f"{base_filename}.pt")
                    try:
                        torch.save(tokens.cpu(), tokens_path)
                    except Exception as e:
                        #print(f"Error saving visual tokens: {str(e)}")
                        pass
                else:
                    print(f"No tokens available for batch {batch_idx}")
                    pass


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # 강제로 distributed 비활성화
    args.launcher = 'none'
    distributed = False

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
        **dict(workers_per_gpu=1),  # worker 수도 줄임
        **dict(samples_per_gpu=1),  # batch size를 1로 고정
        **cfg.data.get('train_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **data_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    
    # Feature Extractor 설정
    feature_extractor = FeatureExtractor(save_path=save_dir)
    feature_extractor.batch_count = 0  # 배치 카운터 초기화
    
    # transformer layer에 대한 hook 등록
    print("\n=== Registering hooks for transformer layers ===")
    
    for name, module in model.named_modules():
        # attention hook 등록 - 이 부분이 중요합니다!
        if 'keypoint_head.tokenpose.transformer_layer1.layers.9.0.fn.fn.to_qkv' in name:
            try:
                hook = module.register_forward_hook(
                    feature_extractor.save_attention(name))
                feature_extractor.hooks.append(hook)
                print(f"Successfully registered attention hook for: {name}")
            except Exception as e:
                print(f"Error registering attention hook for {name}: {e}")
        
        # visual token hook 등록
        if 'keypoint_head.tokenpose.transformer_layer1.layers.9.0.fn.norm' in name:
            try:
                hook = module.register_forward_hook(
                    feature_extractor.save_visual_tokens(name))
                feature_extractor.hooks.append(hook)
                print(f"Successfully registered visual token hook for: {name}")
            except Exception as e:
                print(f"Error registering visual token hook for {name}: {e}")
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = []
        
        # 진행상황 파일 경로를 명확하게 지정
        progress_file = os.path.join(cfg.work_dir, 'attention_progress.txt')
        print(f"\nLooking for progress file at: {progress_file}")

        # 진행상황 파일 확인
        start_idx = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    start_idx = int(f.read().strip())
                print(f"\nFound progress file. Resuming from index: {start_idx}")
            except Exception as e:
                print(f"\nError reading progress file: {str(e)}")
                start_idx = 0
        else:
            print("\nNo progress file found. Starting from beginning.")
        
        try:
            rank, _ = get_dist_info()
            if rank == 0:
                prog_bar = mmcv.ProgressBar(len(data_loader))
                for _ in range(start_idx):
                    prog_bar.update()
            
            for batch_idx, data in enumerate(data_loader):
                if batch_idx < start_idx:
                    continue
                
                try:
                    print(f"\nProcessing batch {batch_idx}...")
                    
                    # 현재 진행상황 저장
                    with open(progress_file, 'w') as f:
                        f.write(str(batch_idx))
                    
                    # forward pass
                    with torch.cuda.amp.autocast():
                        output = model(return_loss=False, **data)
                    outputs.extend(output)
                    
                    # attention maps 저장 - 이 부분 확실히 실행되도록
                    print(f"Available attention maps: {list(feature_extractor.attentions.keys())}")
                    for att_name, attention_list in feature_extractor.attentions.items():
                        if attention_list:  # 리스트가 비어있지 않은 경우에만
                            attention = attention_list[-1]
                            save_path = os.path.join(save_dir, f'{att_name}')
                            print(f"Saving attention map for {att_name}")
                            visualize_attention_maps(attention, save_path, 
                                                  f"{att_name} (Batch {batch_idx})", 
                                                  data=data,
                                                  batch_idx=batch_idx,
                                                  feature_extractor=feature_extractor)
                    
                    if rank == 0:
                        prog_bar.update()
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        print(f"\nBatch {batch_idx}/{len(data_loader)} | GPU Memory: {gpu_memory:.2f} GB")
                    
                    # 메모리 정리
                    torch.cuda.empty_cache()
                    feature_extractor.clear_previous_batch()
                    
                except Exception as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    traceback.print_exc()
                    with open(progress_file, 'w') as f:
                        f.write(str(batch_idx))
                    continue
                
                # 진행상황 저장
                with open(progress_file, 'w') as f:
                    f.write(str(batch_idx + 1))
                
                # 주기적 메모리 정리
                if batch_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            with open(progress_file, 'w') as f:
                f.write(str(batch_idx))
        
        finally:
            feature_extractor.remove_hooks()
            torch.cuda.empty_cache()
            gc.collect()
            
            # 모든 처리가 완료된 경우에만 진행상황 파일 삭제
            if rank == 0 and batch_idx >= len(data_loader) - 1:
                if os.path.exists(progress_file):
                    os.remove(progress_file)

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        
        def process_batch_distributed(model, data, batch_idx):
            with torch.no_grad():
                output = model(return_loss=False, **data)
                
                if get_dist_info()[0] == 0:  # rank 0 프로세스에서만 처리
                    print(f"\nProcessing batch {batch_idx}")
                    
                    # MCT attention maps 처리
                    for att_name, attention_list in feature_extractor.attentions.items():
                        if 'transformer_layer' in att_name:
                            # 현재 배치의 attention map 가져오기
                            attention = attention_list[batch_idx]
                            
                            # 배치 내의 모든 이미지에 대해 처리
                            for img_idx in range(len(data['img'])):
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
