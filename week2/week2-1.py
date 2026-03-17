def sigma(x):
    # ret이라는 변수를 0으로 초기화
    ret = 0
    
    # 매개변수 x로 받은 리스트의 요소를 하나씩 꺼내 x_i에 대입하며 반복
    for x_i in x:
        # 현재의 x_i 값을 ret 변수에 더해줌
        ret += x_i
        
    # 모든 반복이 끝나면 최종 합계인 ret을 반환
    return ret

# 리스트 컴프리헨션을 사용하여 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 리스트를 생성
x = [i + 1 for i in range(10)]

# sigma 함수에 생성한 리스트 x를 전달하고, 그 결과값(55)을 출력
print(sigma(x))