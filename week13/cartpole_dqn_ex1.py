#!/usr/bin/env python
# coding: utf-8

# # 카트폴 게임 마스터하기
# cartpole_dqn.py의 수정본: gym 최신 API(>=0.26) 및 numpy 호환성 문제를 반영

import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# numpy 구버전에서 np.bool8이 제거된 경우를 위한 하위 호환 처리
np.bool8 = np.bool_  # Add alias for backward compatibility

# ### 하이퍼파라미터
# 하이퍼파라미터
EPISODES = 50    # 애피소드 반복횟수 (원본 5에서 50으로 복원)
EPS_START = 0.9  # 학습 시작시 에이전트가 무작위로 행동할 확률
EPS_END = 0.05   # 학습 막바지에 에이전트가 무작위로 행동할 확률
EPS_DECAY = 200  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
GAMMA = 0.8      # 할인계수
LR = 0.001       # 학습률
BATCH_SIZE = 64  # 배치 크기


# ## DQN 에이전트

class DQNAgent:
    def __init__(self):
        # 상태(4차원) -> 행동(2개: 왼쪽/오른쪽)을 출력하는 2층 신경망
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0          # 탐험률(epsilon) 감쇠 계산에 사용
        self.memory = deque(maxlen=10000)  # 경험 재플레이 버퍼 (최대 1만 개)

    def memorize(self, state, action, reward, next_state):
        # 한 스텝의 경험 (s, a, r, s')을 버퍼에 저장
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))
    
    def act(self, state):
        # epsilon-greedy 정책: 초반엔 무작위 탐험, 후반엔 모델 활용
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            # 모델이 예측한 Q값이 높은 행동 선택
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            # 무작위 행동 (0 또는 1)
            return torch.LongTensor([[random.randrange(2)]])
    
    def learn(self):
        # 버퍼에 BATCH_SIZE 미만이면 학습 스킵
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        # 현재 상태에서 선택한 행동의 Q값
        current_q = self.model(states).gather(1, actions)
        # 다음 상태의 최대 Q값 (detach로 타깃 측 그래디언트 차단)
        max_next_q = self.model(next_states).detach().max(1)[0]
        # 벨만 방정식: Q_target = r + gamma * max Q(s', a')
        expected_q = rewards + (GAMMA * max_next_q)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ## 학습 준비하기
# `gym`을 이용하여 `CartPole-v1`환경을 준비하고 앞서 만들어둔 DQNAgent를 agent로 인스턴스화 합니다.
# 자, 이제 `agent` 객체를 이용하여 `CartPole-v1` 환경과 상호작용을 통해 게임을 배우도록 하겠습니다.

env = gym.make('CartPole-v1', render_mode='human')
agent = DQNAgent()
score_history = []  # 에피소드별 생존 스텝 수 기록


# ## 학습 시작

for e in range(1, EPISODES+1):
    # gym>=0.26: reset()이 (state, info) 튜플을 반환하므로 언패킹 필요
    state, _ = env.reset()
    steps = 0
    while True:
        # gym>=0.26: 관측값이 list가 아닌 numpy array일 수 있어 np.array() 변환 필요
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        action = agent.act(state)
        # gym>=0.26: step()이 (obs, reward, terminated, truncated, info) 반환
        next_state, reward, terminated, truncated, info = env.step(action.item())
        # 시간 초과(truncated) 또는 막대 낙하(terminated) 모두 에피소드 종료로 처리
        done = terminated or truncated

        # 게임이 끝났을 경우 마이너스 보상주기
        if done:
            reward = -1  # 막대가 쓰러지면 패널티 부여

        agent.memorize(state, action, reward, next_state)
        agent.learn()

        state = next_state
        steps += 1

        if done:
            print("에피소드:{0} 점수: {1}".format(e, steps))
            score_history.append(steps)
            break


# 에피소드별 점수 추이 시각화
plt.plot(score_history)
plt.ylabel('score')
plt.show()
