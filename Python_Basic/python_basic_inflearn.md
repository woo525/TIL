# 파이썬 기본 문법 정리 - 인프런 강의

### 변수명 정하기
    - 영문과 숫자, _ 로 이루어진다.
    - 대소문자를 구분한다.
    - 문자나, _ 로 시작한다.
    - 특수문자를 사용하면 안된다.(&, % 등)
    - 키워드를 사용하면 안된다.(if, for 등)

### 재귀함수와 스택
```python
# 재귀함수와 스택 

import sys
def DFS(x): # 재귀함수 -> 스택을 활용해 운용: 상태트리를 그려보는 습관
    if x>0:
        DFS(x-1)
        print(x, end=' ')

if __name__=="__main__": # 메인 파트와 재귀함수 구분!
    n=int(input())
    DFS(n) # 깊이 우선 탐색(재귀함수 == DFS)
```

### 데크
```python
# 데크 사용
from collections import deque 
p=deque(p)
p.popleft()
```

### 유용한 스킬
```python
# 유용한 스킬
tmp=board[j][i:i+5] # 리스트 슬라이스: i 부터 (i+5) 앞 원소까지!
if tmp==tmp[::-1] # 리버스 시켜 비교
a=[list(map(int, input().split())) for _ in range(n)] # 이차원 리스트 입력
maxPython=2147000000 # 파이썬 정수 범위 최댓값으로 사용

meeting=[]
for i in range(n):
    s, e=map(int, input().split())
    meeting.append((s, e)) # 튜플 형태로 넣어주기
meeting.sort(key=lambda x : (x[1], x[0])) # 튜플 마지막값 기준으로 정렬
```

### 출력방식
```python
# 출력방식
a, b, c=1, 2, 3
print(a, b, c)
print(a, b, c, sep=', ')
print(a, b, c, sep='')
print(a, end=' ')
print(b, end=' ')
print(c)
```

### 변수입력과 연산자
```python
# 변수입력과 연산자
a, b=map(int, input("숫자를 입력하세요 : ").split()) # 바로 int형으로 형변환
print(a//b) # 몫
print(a**b) # 제곱
```

### 조건문
```python
# 조건문
x=7
if 0<x<10: # 파이썬은 조건문 이렇게 작성 가능
    print("1보다 작은 자연수")
```

### 반복문
```python
# 반복문
a=range(1,11) # 정수 리스트를 만드는 range 함수
print(list(a))
for i in range(10, 0, -2): # 10~2 까지 -2 간격으로
    print(i)
for i in range(1, 11):
    print(i)
    if i>15:
        break
else: # **반복문 안의 if문이 실행되지 않으면 실행**
    print(11)


# 중첩반복문
for i in range(5):
    for j in range(5-i):
        print("*", end=' ')
    print()
```

### 문자열과 내장함수
```python
# 문자열과 내장함수
msg="It is Time"
print(msg.upper())
print(msg.lower())
tmp=msg.upper()
print(tmp.find('T'))
print(tmp.count('T'))
print(msg[:4])
print(len(msg))
for i in range(len(msg)):
    print(msg[i], end=' ')
print()
for x in msg:
    print(x, end=' ')
print()
for x in msg:
    if x.isupper():
        print(x, end=' ')
print()
for x in msg:
    if x.isalpha():
        print(x, end='')
print()
tmp='AZ'
for x in tmp:
    print(ord(x))
```

### 리스트와 내장함수
```python
# 리스트와 내장함수(1)
import random as r

a=[1, 2, 3, 4, 5]
b=list(range(1, 11))
c=a+b
print(c)

a.append(6) # 값 6 맨 뒤 첨가
a.insert(3, 7) # 인덱스 3 자리에 값 7 삽입
a.pop()
a.pop(3) # 인덱스 3 해당값  제거
a.remove(4) # 값 4 찾아서 제거

print(a.index(5)) # 값 5의 인덱스

a=list(range(1, 11)) # 1~10 리스트 생성
print(sum(a)) # 리스트값 총합
print(max(a)) # 최댓값
print(min(a)) # 최솟값
print(min(7, 3, 5)) 
r.shuffle(a) # 리스트 무작위로 섞기
a.sort() # 오름차순 정렬
a.sort(reverse=True) # 내림차순 정렬
a.clear() # 리스트값 모두 삭제


# 리스트와 내장함수(2)
a=[23, 12, 36, 53, 19]
print(a[:3])
print(len(a))
for i in range(len(a)):
    print(a[i], end=' ')
print()
for x in a:
    print(x, end=' ')
print()
for x in enumerate(a):
    print(x, end=' ')
print()

b=(1, 2, 3, 4, 5) # 대괄호: 리스트 // 소괄호: 튜플
print(b[0])
# b[0]=7 # 튜플 값은 이렇게 변경 불가 -> 리스트와의 차이점

for x in enumerate(a): # 인덱스, 값 짝을 지어줌
    print(x[0], x[1])
for index, value in enumerate(a):
    print(index, value)

if all(60>x for x in a): # 모든 경우가 다 참이면 all 함수가 참
    print("yes")
else:
    print("no")

if any(15>x for x in a): # 한가지 경우라도 참이면 any 함수가 참
    print("yes")
else:
    print("no")
```

### 2차원 리스트 생성과 접근
```python
# 2차원 리스트 생성과 접근
a=[0]*3
print(a)

a=[[0]*3 for _ in range(3)] # [0]*3을 3번 반복 -> 2차원 리스트 생성
print(a)

for x in a:
    print(x)

for x in a:
    for y in x:
        print(y, end=' ')
    print()
```

### 함수 만들기
```python
# 함수 만들기
def add(a, b):
    c=a+b
    d=a-b
    return c, d # 튜플 자료형으로 여러 개의 값을 반환 가능
print(add(3, 2))

def isPrime(x):
    for i in range(2, x):
        if(x%i==0):
            return False
    return True
a=[12, 13, 7, 9, 19]
for y in a:
    if isPrime(y):
        print(y, end=' ')
```

### 람다 함수
```python
# 람다 함수
def plus_one(x):
    return x+1
print(plus_one(1))

plus_two=lambda x: x+2
print(plus_two(1))

a=[1, 2, 3]
print(list(map(plus_one, a))) # map(함수, 인자)
print(list(map(lambda x: x+1, a))) # 이러한 형태로 람다함수 활용
```