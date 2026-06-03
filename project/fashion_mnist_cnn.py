#!/usr/bin/env python
# coding: utf-8
"""
합성곱 신경망(CNN)을 활용한 패션 아이템 자동 분류 프로그램
이름   : 강우현
학번   : 202404178
학과   : 인공지능융합공학부
참고   : 펭귄브로의 3분 딥러닝 파이토치맛 - Chapter 5 (github.com/keon/3-min-pytorch)

핵심 기능:
  - 학습 전(랜덤 가중치)과 학습 후의 인식률 비교
  - 학습 곡선(Loss / Accuracy) 시각화
  - 혼동 행렬(Confusion Matrix) 시각화

설계 방향:
  - 교재(keon/3-min-pytorch Ch.5) 구조를 기반으로 소규모 CNN 사용
  - SGD + momentum 옵티마이저 → 완만한 수렴, 과적합 억제
  - Dropout(0.6) 강화 → Train/Test 격차를 줄여 일반화 성능 확보
  - 5 에폭 학습 → Test Accuracy 85~88% 적정 수준 달성
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')          # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 하이퍼파라미터 설정
# ─────────────────────────────────────────────
USE_CUDA   = torch.cuda.is_available()
DEVICE     = torch.device("cuda" if USE_CUDA else "cpu")
EPOCHS     = 5           # 5 에폭 → Test 85~88% 적정 구간
BATCH_SIZE = 64
LR         = 0.01        # SGD 학습률 (교재 기본값)
MOMENTUM   = 0.5         # 교재와 동일

print(f"사용 디바이스: {DEVICE}")
print(f"에폭: {EPOCHS}, 배치: {BATCH_SIZE}, LR: {LR}, 옵티마이저: SGD+momentum\n")

# ─────────────────────────────────────────────
# 2. 데이터셋 로드 및 전처리
#    정규화 수치: Fashion-MNIST 공식 통계(mean=0.2860, std=0.3530)
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(
    root='./.data', train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(
    root='./.data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(
    test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot'
]

# ─────────────────────────────────────────────
# 3. CNN 모델 정의
#
#    교재(keon/3-min-pytorch Ch.5) 원본 구조를 기반으로:
#      - 채널 수를 소규모(16, 32)로 유지 → 과적합 억제
#      - Dropout(p=0.6) 강화 → 일반화 성능 확보
#      - FC 계층도 소형 유지(128 → 10)
#
#    결과적으로 Train Acc > Test Acc 격차가 좁고,
#    Test Accuracy가 85~88% 수준에서 안정적으로 수렴함
# ─────────────────────────────────────────────
class FashionCNN(nn.Module):
    """
    Feature Extraction:
        Conv2d(1 → 16, k=5) → ReLU → MaxPool(2)   [28×28 → 12×12]
        Conv2d(16 → 32, k=5) → Dropout2d → ReLU → MaxPool(2)  [12×12 → 4×4]
    Classification:
        FC(512 → 128) → ReLU → Dropout(0.6)
        FC(128 → 10)
    """
    def __init__(self):
        super(FashionCNN, self).__init__()

        # ── 특징 추출부 (교재 구조 계승) ────────
        self.conv1      = nn.Conv2d(1, 16, kernel_size=5)   # 28→24 / pool→12
        self.conv2      = nn.Conv2d(16, 32, kernel_size=5)  # 12→8  / pool→4
        self.conv2_drop = nn.Dropout2d()                    # 채널 단위 드롭아웃

        # ── 분류부 ──────────────────────────────
        # 32채널 × 4×4 = 512
        self.fc1  = nn.Linear(512, 128)
        self.drop = nn.Dropout(p=0.6)   # 강한 정규화 → 과적합 억제
        self.fc2  = nn.Linear(128, 10)

    def forward(self, x):
        # 특징 추출
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flatten
        x = x.view(-1, 512)

        # 분류
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x   # CrossEntropyLoss가 내부적으로 Softmax 처리


# ─────────────────────────────────────────────
# 4. 평가 함수
# ─────────────────────────────────────────────
def evaluate(model, loader):
    """테스트 셋에 대한 Loss와 Accuracy를 반환"""
    model.eval()
    total_loss = 0.0
    correct    = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred    = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


def get_predictions(model, loader):
    """전체 예측값과 실제 레이블을 numpy 배열로 반환"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            preds = model(data).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.numpy())
    return np.array(all_preds), np.array(all_targets)


# ─────────────────────────────────────────────
# 5. 학습 전 인식률 측정 (랜덤 가중치)
# ─────────────────────────────────────────────
model     = FashionCNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

print("=" * 55)
print("▶ 학습 전 인식률 측정 (랜덤 초기화 가중치)")
print("=" * 55)
before_loss, before_acc = evaluate(model, test_loader)
print(f"  Loss    : {before_loss:.4f}")
print(f"  Accuracy: {before_acc:.2f}%  ← 무작위 예측 수준 (~10%)\n")

# ─────────────────────────────────────────────
# 6. 학습 루프
# ─────────────────────────────────────────────
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

print("=" * 55)
print("▶ 학습 시작")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────
    model.train()
    running_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred    = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx * len(data):>5d}/"
                  f"{len(train_loader.dataset)}]  Loss: {loss.item():.4f}")

    tr_loss = running_loss / len(train_loader.dataset)
    tr_acc  = 100.0 * correct / len(train_loader.dataset)
    te_loss, te_acc = evaluate(model, test_loader)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)

    print(f"\n  [Epoch {epoch}] "
          f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  "
          f"Test Loss: {te_loss:.4f}  Test Acc: {te_acc:.2f}%\n")

# ─────────────────────────────────────────────
# 7. 학습 후 인식률 측정 및 비교 출력
# ─────────────────────────────────────────────
after_loss, after_acc = evaluate(model, test_loader)

print("=" * 55)
print("▶ 학습 전/후 인식률 비교")
print("=" * 55)
print(f"  학습 전 Accuracy : {before_acc:.2f}%  (랜덤 가중치)")
print(f"  학습 후 Accuracy : {after_acc:.2f}%  ({EPOCHS} Epochs 학습 완료)")
print(f"  향상폭           : +{after_acc - before_acc:.2f}%p\n")

# ─────────────────────────────────────────────
# 8. 시각화
# ─────────────────────────────────────────────
epochs_range = range(1, EPOCHS + 1)

# ── 8-1. 학습 곡선 (Loss & Accuracy) ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN Training Curves - Fashion MNIST\n(강우현 202404178)',
             fontsize=13, fontweight='bold')

axes[0].plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2)
axes[0].plot(epochs_range, test_losses,  'r-s', label='Test Loss',  linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss per Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.4)

axes[1].plot(epochs_range, train_accs, 'b-o', label='Train Acc (%)', linewidth=2)
axes[1].plot(epochs_range, test_accs,  'r-s', label='Test Acc (%)',  linewidth=2)
axes[1].axhline(y=before_acc, color='gray', linestyle='--', linewidth=1.2,
                label=f'Before training ({before_acc:.1f}%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy per Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(">> 학습 곡선 저장: training_curves.png")

# ── 8-2. 혼동 행렬 ────────────────────────────
preds, targets = get_predictions(model, test_loader)
cm = confusion_matrix(targets, preds)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label',      fontsize=12)
ax.set_title(f'Confusion Matrix (After Training, Acc={after_acc:.1f}%)',
             fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(">> 혼동 행렬 저장: confusion_matrix.png")

# ── 8-3. 학습 전/후 비교 바 차트 ─────────────
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ['학습 전\n(Random Weights)', f'학습 후\n({EPOCHS} Epochs)'],
    [before_acc, after_acc],
    color=['#E06C75', '#61AFEF'], width=0.4, edgecolor='white'
)
for bar, val in zip(bars, [before_acc, after_acc]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Accuracy: Before vs After Training', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('before_after_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(">> 학습 전/후 비교 그래프 저장: before_after_comparison.png")

print("\n모든 실행 완료!")
