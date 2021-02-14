# CH2: 데이터 다루기
---------------------------------------------------------------------------------

### 2-1: 훈련 세트와 테스트 세트

- 도미와 빙어 데이터 파이썬 리스트로 준비: <http://bit.ly/bream_smelt>

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 여기서 하나의 생선 데이터를 '샘플'이라고 함

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

# 사이킷런의 KNeighborsClassifier 클래스를 임포트하고 모델 객체를 만듦
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

# 리스트의 슬라이싱을 활용, 훈련 세트와 테스트 세트로 분류
# 슬라이싱을 사용할 때는 마지막 인덱스의 원소는 포함되지 않음
# 훈련 세트
train_input = fish_data[:35]
train_target = fish_target[:35]
# 테스트 세트
test_input = fish_data[35:]
test_target = fish_target[35:]

# 이차원 리스트, 리스트의 리스트에서 슬라이싱 연산자를 사용하면 리스트의 리스트 형태로 반환해준다!!

# 훈련 세트로 fit() 호출해 모델 훈련 + 테스트 세트로 score() 호출해 평가
kn = kn.fit(train_input, train_target) # 훈련
kn.score(test_input, test_target) # 평가

# BUT, 이렇게 할 경우 '샘플링 편향' 문제 발생!! 
# 훈련 세트와 테스트 세트를 나누려면 도미와 빙어가 골고루 섞이게 만들어야 함

# 넘파이 라이브러리(대표적인 배열 라이브러리) 임포트
import numpy as np

# 파이썬 리스트를 넘파이 배열로 바꾸기
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# print(input_arr)
# print(input_arr.shape) # 배열의 크기를 알려줌, (샘플 수, 특성 수) 출력! 
 
np.random.seed(42) # 일정한 결과를 얻기 위해 초기에 랜덤 시드를 지정 
index = np.arange(49) # 0~48까지 1씩 증가하는 배열 만들기
np.random.shuffle(index) # 주어진 배열을 무작위로 섞기
# print(index)

# 배열 인덱싱: 1개의 인덱스가 아닌 여러 개의 인덱스로 한 번에 여러 개의 원소를 선택
# 훈련 세트
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
# 테스트 세트
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 훈련 세트와 테스트 세트에 도미와 빙어가 잘 섞여 있는지 산점도 그리기
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 머신러닝 프로그램 다시 만들기 
# fit(): 실행할 때마다 KNeighborsClassifier 클래스의 객체는 이전에 학습한 모든 것을 잃어버림
kn = kn.fit(train_input, train_target) # 훈련
kn.score(test_input, test_target) # 평가

kn.predict(test_input) # 테스트 세트의 예측 결과
test_target # 실제 타깃
```

##### 정리하기

- 공정하게 점수를 메기기 위해서는 훈련에 참여하지 않은 샘플을 사용
	- 훈련 세트: 모델을 훈련 / 테스트 세트: 모델을 평가 

- 지도 학습: 입력과 타깃을 전달한여 모델을 훈련한 다음 새로운 데이터를 예측하는 데 활용 ex) k-최근접 이웃

- 비지도 학습: 타깃 데이터가 없음 -> 입력 데이터에서 어떤 특징을 찾는 데 주로 활용

- 훈련 세트: 훈련 세트 클수록 좋음

- 테스트 세트: 전체 데이터에서 20%, 30%를 테스트 세트로 사용하는 경우가 많음

- numpy

	- seed(): 넘파이에서 난수를 생성하기 위한 정수 초깃값을 지정 -> 초깃값이 같으면 동일한 난수를 뽑을 수 있음
	- arrange(): 일정한 간격의 정수 또는 실수 배열을 만듦 -> 0에서 종료 숫자까지 배열을 만듦, 종료 숫자는 배열에 포함되지 않음
	- suffle: 주어진 배열을 랜덤하게 섞음 -> 다차원 배열의 경우 축(행)에 대해서만 섞음 

---------------------------------------------------------------------------------

### 2-2: 데이터 전처리

- 올바른 결과 도출을 위해서 데이터를 사용하기 전에 데이터 전처리 과정 거침
- 표준점수로 특성의 스케일을 변환하는 방법 알아보자!

- 도미와 빙어 데이터: <http://bit/bream_smelt>

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 넘파이를 활용하여 세련된 방법으로 도미와 빙어 데이터를 준비하자!

# 넘파이 임포트
import numpy as np

# 입력 데이터
# column_stack(): 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결
# 연결할 리스트는 파이썬 튜플로 전달
fish_data = np.column_stack((fish_length, fish_weight))

# 타깃 데이터
# np.ones(), np.zeros(): 1과 0을 채운 배열을 만들어줌
# np.concatenate(): 첫번째 차원을 따라 배열을 연결 -> 튜플로 전달
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# 사이킷런으로 훈련 세트와 테스트 세트 나누기 -> train_test_split()
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
# print(train_input.shape, test_input.shape)
# print(train_target.shape, test_target.shape)

# print(test_target) -> 샘플링 편향이 나타남 -> stratify=fish_target
# 클래스 비율에 맞게 데이터를 나눔 -> 샘플링 편향 방지

# 입력 데이터는 2개의 열이 있는 입력 데이터
# 타깃 데이터는 1차원 배열


## 수상한 도미 한마리
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

print(kn.predict([[25, 150]])) # 왜 빙어로 판단하는 것일까?

# kneighbors(): 주어진 샘플에서 가장 가까운 이웃을 찾아주는 메서드
distances, indexes = kn.kneighbors([[25, 150]])

# 산점도를 그려보자!
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^') # marker 매개변수 모양 지정

# 수상한 도미의 주변 5개의 이웃 그리기
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000)) # 이렇게 그려보면 이해가 됨
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
print(train_target[indexes]) # 가장 가까운 데이터: 빙어4, 도미1
print(distances) 

## 왜 이러한 문제가 발생할까? x축과 y축의 범위가 다르기 때문! == 스케일이 다름!
# plt.xlim((0, 1000)) -> 이놈을 추가!

## 데이터 전처리: 특성값을 일정한 기준으로 맞춰 주어야 함
# ex) 표준점수: 각 특성값이 0에서 표준편차의 몇 배만큼 떨어져 있는지
# 평균을 빼고 표준편차를 나누어 주면 됨
mean = np.mean(train_input, axis=0) # 평균(axis=0: 행을 따라 각 열의 통계값)
std = np.std(train_input, axis=0) # 표준편차
# print(mean, std)
train_scaled = (train_input - mean) / std # *브로드캐스팅*


## 전처리 데이터로 다시 모델 훈련
# 변환된 모델 산점도로 그리기!
new = ([25, 150] - mean) / std # 새로운 샘플도 동일한 비율로 변환
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')

kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std # 마찬가지로 정규화
kn.score(test_scaled, test_target)
print(kn.predict([new])) # 새로운 샘플 판별

# 수정된 새로운 샘플의 주변 5개 샘플 표시
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.show()
``` 

##### 정리하기

- 대부분의 머신러닝 알고리즘은 특성의 스케일이 다르면 잘 작동하지 않음
        - 가장 널리 사용하는 방법: 표준점수
        - 주의할 점: 훈련 세트를 변환한 방식 그대로 테스트 세트를 변환(훈련 세트의 통계값 이용 변환)

- 데이터 전처리: 머신러닝 모델에 데이터를 주입하기 전에 가공하는 단계
- 표준점수: 훈련세트의 스케일을 바꾸는 대표적인 방법 -> 훈련 세트의 평균과 표준편차로 테스트 세트를 바꿔야 함
- 브로드캐스팅: 크기가 다른 넘파이 배열에서 자동으로 사칙 연산을 모든 행이나 열로 확장하여 수행하는 기능

- train_test_split(): 훈련 데이터를 훈련 세트와 테스트 세트로 나누는 함수
        - stratify 매개변수에 클래스 레이블이 담긴 배열(일반적으로 타깃 데이터)을 전달하면 클래스 비율에 맞게 훈련 세트와 테스트 세트를 분할
- kneighbors(): k-최근접 이웃 갯체의 메서드, 입력한 데이터에 가장 가까운 이웃을 찾아 거리와 이웃 샘플의 인덱스를 반환  

---------------------------------------------------------------------------------

