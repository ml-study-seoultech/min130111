import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import numpy as np

class DamagedImageDataset(Dataset):
    def __init__(self, csv_path, num_samples=None):
        """
        Args:
            csv_path: train.csv 파일 경로
            num_samples: 테스트용 샘플 수 (None이면 전체 데이터 사용)
        """
        # CSV 파일 읽기
        self.data = pd.read_csv(csv_path)
        if num_samples:
            self.data = self.data.head(num_samples)
            
        # 기본 디렉토리 경로 (csv 파일이 있는 디렉토리)
        self.root_dir = os.path.dirname(csv_path)
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            # transforms.Grayscale(),  # 흑백 이미지
            transforms.ToTensor(),   # [0, 255] -> [0, 1]
            transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]
        ])
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 경로 가져오기
        input_path = os.path.join(self.root_dir, 
                                self.data.iloc[idx]['input_image_path'].replace('./', ''))
        gt_path = os.path.join(self.root_dir, 
                              self.data.iloc[idx]['gt_image_path'].replace('./', ''))
        
        # 이미지 로드
        input_img = Image.open(input_path)
        gt_img = Image.open(gt_path)
        
        # 전처리 적용
        input_tensor = self.transform(input_img)
        gt_tensor = self.transform(gt_img)
        
        return input_tensor, gt_tensor

# def create_random_mask(size=256, mask_size=128):
#     """중앙에 랜덤한 마스크 생성"""
#     mask = torch.ones((size, size))
#     x = np.random.randint(0, size - mask_size)
#     y = np.random.randint(0, size - mask_size)
#     mask[x:x+mask_size, y:y+mask_size] = 0
#     return mask 