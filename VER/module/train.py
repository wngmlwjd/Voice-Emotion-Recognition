import torch
import os
import shutil

def train_model(train_dataloader, model, criterion, optimizer, model_path, num_epochs=10, save_interval=1, batch_interval=10):
    device = torch.device("mps")
    model = model.to(device)
    
    # 저장 디렉토리 생성
    # os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started.")
        
        for batch_idx, (features, labels) in enumerate(train_dataloader, start=1):
            features, labels = features.to(device), labels.to(device)
            
            # 모델에 입력
            features = features.permute(1, 0, 2)  # (seq_len, batch, features), Transformer의 입력 차원에 맞게 차원 변환
            labels = labels.long()
            
            optimizer.zero_grad() # 경사도 초기화
            
            # 예측
            outputs = model(features)
            
            # 디버깅
            predicted_labels = torch.argmax(outputs, dim=1)
            print(f"[DEBUG] Predicted labels: {predicted_labels.tolist()}")
            print(f"[DEBUG] Actual Labels: {labels.tolist()}")
            print()
            
            # 손실 계산
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 현재 배치 상태 출력
            if batch_idx % batch_interval == 0:  # 10번째 배치마다 출력
                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
                
            # 배치마다 모델 덮어쓰기 저장
            if batch_idx % save_interval == 0:
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                    
                torch.save(model, model_path)
                print(f"Model saved")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    
    shutil.copy(model_path, LATEST_MODEL_PATH)
    print(f"Latest model saved to {LATEST_MODEL_PATH}")