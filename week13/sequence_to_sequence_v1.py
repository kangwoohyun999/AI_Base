#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq 기계 번역
# 이번 프로젝트에선 임의로 Seq2Seq 모델을 아주 간단화 시켰습니다.
# 한 언어로 된 문장을 다른 언어로 된 문장으로 번역하는 덩치가 큰 모델이 아닌
# 영어 알파벳 문자열("hello")을 스페인어 알파벳 문자열("hola")로 번역하는 Mini Seq2Seq 모델을 같이 구현해 보겠습니다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# 학습 데이터 정의
training_pairs = [ # 입력 문자열과 대상 타깃 번역 문자열 쌍 리스트 구성
    ("I go to bed.", "Me voy a la cama."),
    ("I want to sleep.", "Quiero dormir."),
    ("I play a game.", "Juego un juego.")
]

# 문자 단위 처리를 위해 전체 ASCII 범위를 어휘 크기로 사용
vocab_size = 256  # 총 아스키 코드 개수

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.n_layers = 2          # 인코더/디코더 GRU 레이어 수
        self.hidden_size = hidden_size
        # 문자 인덱스(ASCII)를 hidden_size 차원 벡터로 변환
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # 입력 시퀀스를 문맥 벡터로 압축하는 GRU 인코더
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers, dropout=0.2)
        # 문맥 벡터로부터 출력 시퀀스를 생성하는 GRU 디코더
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers, dropout=0.2)
        # 어텐션: 디코더 출력과 인코더 출력을 결합하기 위한 선형 변환
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        # 디코더 출력 + 컨텍스트 -> 어휘 크기의 로짓으로 변환
        self.project = nn.Linear(hidden_size * 2, vocab_size)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        # Xavier 초기화: 가중치 행렬의 분산을 레이어 크기에 맞게 조정
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, inputs, targets):
        batch_size = 1
        seq_len = inputs.size(0)
        
        # 인코더: 입력 문자 시퀀스를 처리해 은닉 상태(문맥 벡터)를 생성
        initial_state = self._init_state(batch_size)
        embedding = self.embedding(inputs).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        
        # 디코더: 인코더의 마지막 은닉 상태를 초기 상태로 사용
        decoder_state = encoder_state
        # 'V'(ASCII 86)를 디코더 시작 토큰(SOS)으로 사용
        decoder_input = torch.LongTensor([ord('V')]).to(inputs.device)
        
        outputs = []
        for i in range(targets.size(0)):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # 어텐션 계산: 현재 디코더 출력과 모든 인코더 출력 간의 유사도 계산
            decoder_output_reshaped = decoder_output.squeeze(1)
            encoder_output_reshaped = encoder_output.squeeze(1)
            
            # 내적으로 어텐션 점수 계산 후 소프트맥스로 정규화
            attention_scores = torch.matmul(decoder_output_reshaped, encoder_output_reshaped.transpose(0, 1))
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # 어텐션 가중치로 인코더 출력의 가중 평균(컨텍스트 벡터) 계산
            context = torch.matmul(attention_weights, encoder_output_reshaped)
            context = context.unsqueeze(1)
            
            # 디코더 출력과 컨텍스트 벡터를 이어붙여 최종 어휘 로짓 생성
            combined = torch.cat([decoder_output, context], dim=-1)
            projection = self.project(combined)
            
            outputs.append(projection)
            # 교사 강요(teacher forcing): 예측값 대신 실제 정답 토큰을 다음 입력으로 사용
            decoder_input = torch.LongTensor([targets[i]]).to(inputs.device)
        
        outputs = torch.stack(outputs).squeeze()
        return outputs
    
    def translate(self, input_text, max_length=20):
        # 학습 없이 추론만 수행하는 번역 함수 (teacher forcing 없음)
        input_seq = torch.LongTensor(list(map(ord, input_text)))
        print(f"입력 시퀀스: {input_seq}")
        
        # 인코더: 입력 문장을 문맥 벡터로 인코딩
        initial_state = self._init_state(1)
        embedding = self.embedding(input_seq).unsqueeze(1)
        encoder_output, encoder_state = self.encoder(embedding, initial_state)
        
        # 디코더 초기화
        decoder_state = encoder_state
        decoder_input = torch.LongTensor([ord('V')])  # SOS 토큰
        
        translated_text = []
        last_char = None
        repeat_count = 0
        
        for i in range(max_length):
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            
            # 어텐션 적용
            decoder_output_reshaped = decoder_output.squeeze(1)
            encoder_output_reshaped = encoder_output.squeeze(1)
            
            attention_scores = torch.matmul(decoder_output_reshaped, encoder_output_reshaped.transpose(0, 1))
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            context = torch.matmul(attention_weights, encoder_output_reshaped)
            context = context.unsqueeze(1)
            
            combined = torch.cat([decoder_output, context], dim=-1)
            projection = self.project(combined)
            
            # 소프트맥스 적용 및 예측
            probs = F.softmax(projection.squeeze(0), dim=-1)
            values, indices = torch.topk(probs, k=1)
            next_char = indices.item()
            
            print(f"Step {i}:")
            print(f"  Projection shape: {projection.size()}")
            print(f"  Values: {values.detach().tolist()}")
            print(f"  Indices: {indices.detach().tolist()}")
            print(f"  Predicted char: {chr(next_char)} (ASCII: {next_char})")
            
            # 출력 불가능한 ASCII 범위(제어 문자 등)이면 생성 종료
            if next_char < 32 or next_char > 126:  # 유효하지 않은 ASCII 문자
                break
                
            # 동일 문자가 3회 이상 반복되면 루프 탈출 (반복 출력 방지)
            if next_char == last_char:
                repeat_count += 1
                if repeat_count > 2:  # 같은 문자가 3번 이상 반복되면 종료
                    break
            else:
                repeat_count = 0
                last_char = next_char
            
            # 마침표가 나오면 문장 종료로 간주
            if len(translated_text) > 0 and next_char == ord('.'):
                break
                
            translated_text.append(chr(next_char))
            # 자기 회귀 방식: 직전 예측 문자를 다음 스텝의 입력으로 사용
            decoder_input = torch.LongTensor([next_char])
        
        result = ''.join(translated_text)
        print(f"최종 번역 결과: {result}")
        return result
    
    def _init_state(self, batch_size=1):
        # n_layers * batch_size * hidden_size 크기의 영벡터 은닉 상태 생성
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()

# 모델 초기화
seq2seq = Seq2Seq(vocab_size, 64)  # 은닉층 크기 증가
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)
# 검증 손실이 개선되지 않으면 학습률을 자동으로 줄이는 스케줄러
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 학습
log = []
for epoch in range(1000):  # 에포크 수 증가
    total_loss = 0
    for x_text, y_text in training_pairs:
        # 문자열을 ASCII 코드 텐서로 변환
        x = torch.LongTensor(list(map(ord, x_text)))
        y = torch.LongTensor(list(map(ord, y_text)))
        
        prediction = seq2seq(x, y)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        # 그래디언트 폭발 방지를 위한 클리핑 (최대 노름 1.0)
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(training_pairs)
    scheduler.step(avg_loss)
    log.append(avg_loss)
    
    if epoch % 100 == 0:
        print(f"\nEpoch: {epoch}, Average Loss: {avg_loss}")
        # 각 학습 예제에 대한 예측 출력
        for x_text, y_text in training_pairs:
            x = torch.LongTensor(list(map(ord, x_text)))
            y = torch.LongTensor(list(map(ord, y_text)))
            prediction = seq2seq(x, y)
            _, top1 = prediction.data.topk(1, 1)
            print(f"Input: {x_text}")
            print(f"Prediction: {[chr(c) for c in top1.squeeze().tolist()]}")

# 손실 그래프 출력
plt.plot(log)
plt.ylabel('cross entropy loss')
plt.show()

# 번역 테스트
print("\n번역 테스트:")
test_texts = ["I go to bed", "I want to sleep", "I play a game"]
for text in test_texts:
    print(f"\n입력: {text}")
    translated_text = seq2seq.translate(text)
    print(f"번역: {translated_text}")
