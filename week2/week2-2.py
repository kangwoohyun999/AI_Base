import numpy as np  # 수치 계산을 위해 numpy 라이브러리를 불러옴

def argmax(target_func, X):
    # 최댓값을 추적하기 위한 변수. 초기값은 가장 작은 값인 '음의 무한대'로 설정
    max_val = -np.inf
    # 최댓값을 만드는 x 값을 저장할 변수. 초기값은 아무것도 없는 None
    ret = None
    
    # 입력 리스트 X에 있는 원소를 하나씩 꺼냄
    for x in X:
        # 현재 원소 x를 함수 target_func에 넣어 결과값(함숫값)을 계산
        val = target_func(x)
        
        # 만약 방금 계산한 값이 기존에 알고 있던 최댓값(max_val)보다 크다면,
        if val > max_val:
            # 최댓값을 현재의 새로운 값으로 업데이트
            max_val = val
            # 그때의 입력값 x를 결과 변수 ret에 저장
            ret = x
            
    # 모든 반복이 끝난 후, 최댓값을 함숫값으로 가지는 입력값(x)을 반환
    return ret

# 1부터 10까지의 정수가 담긴 리스트 [1, 2, ..., 10]을 생성
X = [i + 1 for i in range(10)]

# 목적 함수 f(x)를 정의. 여기서는 -log(x)를 계산
def f(x):
    return -np.log(x)

# f(x)를 최대로 만드는 X 내의 원소를 찾아 출력
# -log(x)는 감소함수이므로 x가 가장 작을 때(1일 때) 최댓값
print(argmax(f, X))