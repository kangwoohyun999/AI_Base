#!/usr/bin/env python
# coding: utf-8

# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

# 하이퍼파라미터
BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)

# 데이터 로딩하기
print("데이터 로딩중...")
# 영어 기본 토크나이저: 공백/구두점 기준으로 단어 분리
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    # 데이터셋을 순회하며 각 리뷰를 토크나이즈한 결과를 제너레이터로 반환
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# 훈련 데이터에서 어휘 사전 구축; 특수 토큰을 앞쪽에 배치
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
# 사전에 없는 단어는 <unk>로 처리
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    # 텍스트 문자열을 어휘 인덱스 리스트로 변환
    return vocab(tokenizer(x))

def label_pipeline(x):
    # IMDB 레이블(1/2)을 0-based 인덱스(0/1)로 변환
    return int(x) - 1

def process_batch(batch, batch_size):
    # 가변 길이 시퀀스를 동일 길이로 패딩하여 텐서로 묶음
    label_list, text_list, lengths = [], [], []
    for i, (_label, _text) in enumerate(batch):
        if i >= batch_size:
            break
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        # 빈 텍스트 방어 처리: 최소 1개의 패딩 토큰 삽입
        if len(processed_text) == 0:
            processed_text = torch.tensor([vocab["<pad>"]], dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    
    if not text_list:  # 빈 배치 처리
        return None, None, None
    
    # 배치 내 최대 길이에 맞춰 짧은 시퀀스를 <pad>로 채움
    max_length = max(lengths)
    padded_text_list = []
    for text in text_list:
        if len(text) < max_length:
            padding = torch.full((max_length - len(text),), vocab["<pad>"], dtype=torch.int64)
            padded_text = torch.cat([text, padding])
        else:
            padded_text = text
        padded_text_list.append(padded_text)
    
    text_tensor = torch.stack(padded_text_list)
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    
    return label_tensor.to(DEVICE), text_tensor.to(DEVICE), lengths_tensor.to(DEVICE)

def train(model, optimizer, train_iter, batch_size):
    model.train()
    total_loss = 0
    batch = []
    progress_bar = tqdm(desc='Training')
    
    for i, data in enumerate(train_iter):
        batch.append(data)
        if len(batch) == batch_size:
            label, text, lengths = process_batch(batch, batch_size)
            if label is not None:
                optimizer.zero_grad()
                logit = model(text, lengths)
                loss = F.cross_entropy(logit, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
            batch = []
    
    # 마지막 배치 처리 (batch_size로 나누어 떨어지지 않는 나머지)
    if batch:
        label, text, lengths = process_batch(batch, batch_size)
        if label is not None:
            optimizer.zero_grad()
            logit = model(text, lengths)
            loss = F.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})

def evaluate(model, test_iter, batch_size):
    model.eval()
    corrects, total_loss = 0, 0
    total_samples = 0
    batch = []
    progress_bar = tqdm(desc='Evaluating')
    
    with torch.no_grad():  # 평가 시에는 그래디언트 계산 불필요
        for i, data in enumerate(test_iter):
            batch.append(data)
            if len(batch) == batch_size:
                label, text, lengths = process_batch(batch, batch_size)
                if label is not None:
                    logit = model(text, lengths)
                    # reduction='sum': 배치 평균이 아닌 합산으로 전체 평균을 별도 계산
                    loss = F.cross_entropy(logit, label, reduction='sum')
                    total_loss += loss.item()
                    corrects += (logit.max(1)[1] == label).sum().item()
                    total_samples += len(label)
                    progress_bar.update(1)
                    progress_bar.set_postfix({'accuracy': f'{100.0 * corrects / total_samples:.2f}%'})
                batch = []
        
        # 마지막 배치 처리
        if batch:
            label, text, lengths = process_batch(batch, batch_size)
            if label is not None:
                logit = model(text, lengths)
                loss = F.cross_entropy(logit, label, reduction='sum')
                total_loss += loss.item()
                corrects += (logit.max(1)[1] == label).sum().item()
                total_samples += len(label)
                progress_bar.update(1)
                progress_bar.set_postfix({'accuracy': f'{100.0 * corrects / total_samples:.2f}%'})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_accuracy = 100.0 * corrects / total_samples if total_samples > 0 else 0
    return avg_loss, avg_accuracy

vocab_size = len(vocab)
n_classes = 2  # 긍정(1) / 부정(0)

print("[단어수]: %d [클래스] %d" % (vocab_size, n_classes))


class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers
        # 단어 인덱스를 embed_dim 차원의 밀집 벡터로 변환
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        # GRU: 임베딩 벡터를 순서대로 처리해 시퀀스 문맥 정보를 은닉 상태에 누적
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        # GRU 마지막 은닉 상태 -> 클래스 로짓으로 변환하는 분류 헤드
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, text, lengths):
        embedded = self.embed(text)
        # 패딩을 제외하고 실제 길이만큼만 GRU에 입력 (효율적 계산)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # 최상위 레이어의 마지막 은닉 상태를 리뷰 전체의 압축 표현으로 사용
        hidden = self.dropout(hidden[-1])
        return self.out(hidden)


train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = None
for e in range(1, EPOCHS+1):
    print(f'\nEpoch {e}/{EPOCHS}')
    train(model, optimizer, train_iter, BATCH_SIZE)
    val_loss, val_accuracy = evaluate(model, test_iter, BATCH_SIZE)
    print(f"[Epoch {e}] Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
    
    # 검증 손실이 개선될 때만 모델 가중치 저장 (조기 종료 대용)
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

# 가장 좋은 가중치를 불러와 최종 테스트 수행
model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter, BATCH_SIZE)
print(f'\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')
