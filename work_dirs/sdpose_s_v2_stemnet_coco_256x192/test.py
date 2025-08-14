import torch
import os
import glob
from itertools import combinations

def compare_pt_files(directory):
    # .pt 파일들의 경로 가져오기
    pt_files = sorted(glob.glob(os.path.join(directory, '*.pt')))
    
    print(f"\nComparing {len(pt_files)} .pt files in {directory}")
    
    # 모든 가능한 파일 쌍에 대해 비교
    for file1_path, file2_path in combinations(pt_files, 2):
        try:
            # 텐서 로드
            tensor1 = torch.load(file1_path)
            tensor2 = torch.load(file2_path)
            
            # 텐서 비교
            if torch.equal(tensor1, tensor2):
                print(f"\n⚠️ Identical tensors found:")
                print(f"File 1: {os.path.basename(file1_path)}")
                print(f"File 2: {os.path.basename(file2_path)}")
            
            # 메모리 정리
            del tensor1, tensor2
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError comparing {file1_path} and {file2_path}:")
            print(str(e))

if __name__ == "__main__":
    # 비교할 디렉토리 경로
    directory = "./attention_maps/mct/visual_tokens"
    
    if os.path.exists(directory):
        compare_pt_files(directory)
    else:
        print(f"Directory not found: {directory}")