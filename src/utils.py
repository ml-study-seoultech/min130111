import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

def save_images(input_images, predicted_images, target_images, epoch, save_dir='results'):
    """학습 중인 모델의 예측 결과를 시각화하여 저장
    
    Args:
        input_images: 손상된 입력 이미지 (B x C x H x W)
        predicted_images: 모델이 예측한 이미지 (B x C x H x W)
        target_images: 원본 타겟 이미지 (B x C x H x W)
        epoch: 현재 에포크
        save_dir: 결과물을 저장할 디렉토리
    """
    # [-1, 1] 범위의 텐서를 [0, 1] 범위로 변환
    input_images = (input_images + 1) / 2
    predicted_images = (predicted_images + 1) / 2
    target_images = (target_images + 1) / 2
    
    # 결과 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 배치의 첫 4개 이미지만 시각화
    n_images = min(4, input_images.size(0))
    
    plt.figure(figsize=(15, 5))
    for i in range(n_images):
        # 입력 이미지
        plt.subplot(3, n_images, i + 1)
        plt.imshow(input_images[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Input')
        
        # 예측 이미지
        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(predicted_images[i].detach().cpu().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Predicted')
        
        # 타겟 이미지
        plt.subplot(3, n_images, i + 1 + 2*n_images)
        plt.imshow(target_images[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Target')
    
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close() 