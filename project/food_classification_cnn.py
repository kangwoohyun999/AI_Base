#!/usr/bin/env python
# coding: utf-8
"""
=======================================================
  CNN을 활용한 한국 음식 자동 분류 프로그램
  이름 : 강우현
  학번 : 202404178
  학과 : 인공지능융합공학부
  참고 : keon/3-min-pytorch Chapter 5

-------------------------------------------------------
  실행 순서:
    1. food_crawling.py 먼저 실행 → food_dataset/ 자동 생성
    2. 이 파일(food_classification_cnn.py) 실행

  📁 food_dataset/ 폴더는 두 파일과 같은 위치에 생성됩니다.
     project/
     ├── food_crawling.py
     ├── food_classification_cnn.py
     └── food_dataset/
         ├── train/ (김치찌개, 된장찌개, 비빔밥, 삼겹살, 라면)
         └── test/  (김치찌개, 된장찌개, 비빔밥, 삼겹살, 라면)
=======================================================
"""

import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ──────────────────────────────────────────────────
# 한글 폰트: OS에 설치된 폰트 파일을 직접 탐색해 등록
# → 폰트명 하드코딩 없이 Windows / macOS / Linux 모두 동작
# ──────────────────────────────────────────────────
def find_korean_font():
    candidates = []

    if platform.system() == "Windows":
        windir    = os.environ.get("WINDIR", "C:/Windows")
        font_dir  = os.path.join(windir, "Fonts")
        for name in ["malgun.ttf", "malgunbd.ttf", "NanumGothic.ttf",
                     "gulim.ttc", "batang.ttc"]:
            candidates.append(os.path.join(font_dir, name))

    elif platform.system() == "Darwin":
        dirs  = ["/Library/Fonts", "/System/Library/Fonts",
                 os.path.expanduser("~/Library/Fonts")]
        names = ["AppleGothic.ttf", "AppleSDGothicNeo.ttc",
                 "NanumGothic.ttf", "NanumGothic.otf"]
        candidates = [os.path.join(d, n) for d in dirs for n in names]

    else:  # Linux
        search_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
        ]
        target_names = ["NanumGothic.ttf", "NanumBarunGothic.ttf",
                        "NotoSansCJK-Regular.ttc"]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if any(t.lower() in f.lower() for t in target_names):
                        candidates.append(os.path.join(root, f))

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

_font_path = find_korean_font()
if _font_path:
    fm.fontManager.addfont(_font_path)
    _prop = fm.FontProperties(fname=_font_path)
    plt.rcParams["font.family"] = _prop.get_name()
    print(f"한글 폰트 로드: {os.path.basename(_font_path)}")
else:
    print("[경고] 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")

plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────────
# ★ 경로 설정
#   이 파일(__file__)이 있는 폴더 기준으로 경로를 자동 계산
#   → food_crawling.py가 생성한 food_dataset/을 바로 읽음
# ──────────────────────────────────────────────────
_BASE     = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(_BASE, "food_dataset", "train")
TEST_DIR  = os.path.join(_BASE, "food_dataset", "test")

# 그래프 저장 위치도 같은 폴더
SAVE_DIR  = _BASE

# ──────────────────────────────────────────────────
# ★ 하이퍼파라미터
# ──────────────────────────────────────────────────
IMG_SIZE   = 64
EPOCHS     = 5
BATCH_SIZE = 16
LR         = 0.01
MOMENTUM   = 0.5

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")
print(f"사용 디바이스: {DEVICE}")
print(f"데이터 경로: {_BASE}")
print(f"에폭: {EPOCHS}  배치: {BATCH_SIZE}  LR: {LR}  옵티마이저: SGD+momentum\n")


# ──────────────────────────────────────────────────
# 1. 데이터 전처리 및 로드
# ──────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=test_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f"클래스 ({NUM_CLASSES}개): {CLASS_NAMES}")
print(f"학습 이미지: {len(train_dataset)}장  |  테스트 이미지: {len(test_dataset)}장\n")


# ──────────────────────────────────────────────────
# 2. CNN 모델 정의 (교재 keon/3-min-pytorch Ch.5 계승)
# ──────────────────────────────────────────────────
class FoodCNN(nn.Module):
    def __init__(self, num_classes):
        super(FoodCNN, self).__init__()
        self.conv1      = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2      = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        fc_in = self._get_fc_size()
        self.fc1  = nn.Linear(fc_in, 128)
        self.drop = nn.Dropout(p=0.6)
        self.fc2  = nn.Linear(128, num_classes)

    def _get_fc_size(self):
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        x = F.relu(F.max_pool2d(self.conv1(dummy), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# ──────────────────────────────────────────────────
# 3. 평가 함수
# ──────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            correct += output.argmax(1).eq(target).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n

def get_predictions(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            p = model(data.to(DEVICE)).argmax(1).cpu().numpy()
            preds.extend(p)
            targets.extend(target.numpy())
    return np.array(preds), np.array(targets)


# ──────────────────────────────────────────────────
# 4. 학습 전 인식률
# ──────────────────────────────────────────────────
model     = FoodCNN(NUM_CLASSES).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

print("=" * 55)
print("▶ 학습 전 인식률 측정 (랜덤 초기화 가중치)")
print("=" * 55)
before_loss, before_acc = evaluate(model, test_loader)
print(f"  Loss    : {before_loss:.4f}")
print(f"  Accuracy: {before_acc:.2f}%  ← 무작위 예측 수준\n")


# ──────────────────────────────────────────────────
# 5. 학습 루프
# ──────────────────────────────────────────────────
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []

print("=" * 55)
print("▶ 학습 시작")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct = 0.0, 0

    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()

    tr_loss = running_loss / len(train_loader.dataset)
    tr_acc  = 100.0 * correct / len(train_loader.dataset)
    te_loss, te_acc = evaluate(model, test_loader)

    train_losses.append(tr_loss); test_losses.append(te_loss)
    train_accs.append(tr_acc);    test_accs.append(te_acc)

    print(f"  [Epoch {epoch}/{EPOCHS}] "
          f"Train {tr_acc:.1f}%  Test {te_acc:.1f}%  Loss {te_loss:.4f}")


# ──────────────────────────────────────────────────
# 6. 학습 전/후 비교
# ──────────────────────────────────────────────────
after_loss, after_acc = evaluate(model, test_loader)

print("\n" + "=" * 55)
print("▶ 학습 전/후 인식률 비교")
print("=" * 55)
print(f"  학습 전 Accuracy : {before_acc:.2f}%  (랜덤 가중치)")
print(f"  학습 후 Accuracy : {after_acc:.2f}%  ({EPOCHS} Epochs 학습)")
print(f"  향상폭           : +{after_acc - before_acc:.2f}%p\n")

preds, targets_arr = get_predictions(model, test_loader)
print("▶ 클래스별 분류 성능")
print(classification_report(targets_arr, preds, target_names=CLASS_NAMES))


# ──────────────────────────────────────────────────
# 7. 시각화 저장 (이 파일과 같은 폴더에 저장)
# ──────────────────────────────────────────────────
er = range(1, EPOCHS + 1)

# 7-1. 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("한국 음식 CNN 학습 곡선  |  강우현 202404178", fontsize=12, fontweight="bold")

axes[0].plot(er, train_losses, "b-o", label="학습 손실",   linewidth=2)
axes[0].plot(er, test_losses,  "r-s", label="테스트 손실", linewidth=2)
axes[0].set_title("에폭별 손실(Loss)")
axes[0].set_xlabel("에폭"); axes[0].set_ylabel("손실")
axes[0].legend(); axes[0].grid(alpha=0.4)

axes[1].plot(er, train_accs, "b-o", label="학습 정확도 (%)",   linewidth=2)
axes[1].plot(er, test_accs,  "r-s", label="테스트 정확도 (%)", linewidth=2)
axes[1].axhline(y=before_acc, color="gray", linestyle="--",
                label=f"학습 전 ({before_acc:.1f}%)")
axes[1].set_title("에폭별 정확도")
axes[1].set_xlabel("에폭"); axes[1].set_ylabel("정확도 (%)")
axes[1].legend(); axes[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print(">> training_curves.png 저장")

# 7-2. 혼동 행렬
cm = confusion_matrix(targets_arr, preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_xlabel("예측 레이블", fontsize=11)
ax.set_ylabel("실제 레이블", fontsize=11)
ax.set_title(f"혼동 행렬  (테스트 정확도 = {after_acc:.1f}%)", fontweight="bold")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print(">> confusion_matrix.png 저장")

# 7-3. 학습 전/후 바 차트
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["학습 전\n(랜덤 가중치)", f"학습 후\n({EPOCHS} 에폭)"],
              [before_acc, after_acc], color=["#E06C75", "#61AFEF"], width=0.4)
for b, v in zip(bars, [before_acc, after_acc]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_ylim(0, 110); ax.set_ylabel("정확도 (%)")
ax.set_title("학습 전/후 인식률 비교", fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "before_after.png"), dpi=150, bbox_inches="tight")
plt.close()
print(">> before_after.png 저장")

print("\n모든 실행 완료!")
print(f"결과 저장 위치: {SAVE_DIR}")
