import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import ContextEncoder
from dataset import DamagedImageDataset
from utils import save_images

def train(model, train_loader, criterion, optimizer, device, num_epochs=20):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (damaged, original) in enumerate(train_loader):
            # 데이터를 GPU로 이동
            damaged = damaged.to(device)
            original = original.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(damaged)
            loss = criterion(output, original)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 매 50 배치마다 또는 에포크 마지막에 이미지 저장
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                save_images(damaged, output, original, 
                          epoch=epoch, 
                          save_dir='results')
            
            # 진행상황 출력
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터셋 로드 (num_samples=None이면 전체 데이터 사용, 샘플 개수 지정)
    dataset = DamagedImageDataset('data/raw/train.csv', num_samples= 10)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # 모델 초기화
    model = ContextEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 학습 시작
    train(model, train_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main() 