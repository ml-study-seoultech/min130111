import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from model import ContextEncoder
from dataset import DamagedImageDataset
from utils import save_images

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
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
    
    # 1000장의 데이터 사용
    dataset = DamagedImageDataset('data/raw/train.csv', num_samples=1000)
    
    # 학습/검증 데이터 분할 (90% 학습, 10% 검증)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # 배치 크기를 32로 설정
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f'Total samples: 1000')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # 모델 초기화
    model = ContextEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 학습 시작
    train(model, train_loader, val_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main() 