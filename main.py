import streamlit as st
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import os

# SimpleNN 정의 (기존과 동일)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 데이터 로드
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델, 손실 함수, 옵티마이저 초기화
if 'model' not in st.session_state:
    st.session_state.model = SimpleNN()
    st.session_state.optimizer = optim.SGD(st.session_state.model.parameters(), lr=0.01)
    st.session_state.epoch_count = 0

model = st.session_state.model
optimizer = st.session_state.optimizer
criterion = nn.CrossEntropyLoss()

# 학습 버튼
if st.button("5 epoch 학습 시작"):
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.session_state.epoch_count += 1
        st.write(f"Epoch {st.session_state.epoch_count} Loss: {total_loss:.4f}")
    st.success("5 epoch 학습 완료!")
    # 모델 상태 저장 (선택적)
    torch.save(model.state_dict(), 'model.pth')

# 테스트
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_img, test_label = test_dataset[0]
st.image(test_img.squeeze().numpy(), caption=f"정답: {test_label}")

model.eval()
with torch.no_grad():
    logits = model(test_img.unsqueeze(0))
    pred = torch.argmax(logits, dim=1).item()
st.write(f"모델 예측: {pred}")
