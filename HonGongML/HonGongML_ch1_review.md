# CH1: 나의 첫 머신러닝
---------------------------------------------------------------------------------

### 1-1: 인공지능과 머신러닝, 딥러닝

- 인공지능: 사람처럼 학습하고 추론할 수 있는 지능을 가진 컴퓨터 시스템을 만드는 기술 
	- 강인공지능 VS 약인공지능 

- 머신러닝: 규칙을 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야
	- 사이킷런이 대표적인 라이브러리

- 딥러닝: 많은 머신러닝 알고리즘 중에 인공 신경망을 기반으로 한 방법들을 통칭
	- 텐서플로와 파이토치가 대표적인 라이브러리 

---------------------------------------------------------------------------------

### 1-2: 코랩과 주피터 노트북

- 코랩: 웹 브라우저 기반의 파이썬 코드 실행 환경

- 노트북: 코랩의 프로그램 작성 단위이며 일반 프로그램 파일과 달리 대화식으로 프로그램을 만들 수 있기 때문에 데이터분석이나 교육에 매우 적합
	- 코드, 코드의 실행 결과, 문서를 모두 저장하여 보관

- 마크다운 언어 길라잡이: <https://gist.github.com/ihoneymon/652be052a0727ad59601>

---------------------------------------------------------------------------------

### 1-3: 마켓과 머신러닝

##### 가장 간단한 머신러닝 알고리즘 중 하나인 k-최근접 이웃을 사용하여 2개의 종류를 분류하는 머신러닝 모델을 훈련

- 도미 데이터: <http://bit.ly/bream_list>
- 빙어 데이터: <http://bit.ly/smelt_list>

```python
# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

- 특성(feature): 각 도미의 길이와 무게

- 첫 번째 머신러닝 프로그램 
	
	- 사이킷런 머신러닝 패키지를 사용하려면 각 특성의 리스트를 세로방향으로 늘어뜨린 2차원 리스트를 만들어야 함

	- zip() 함수는 나열된 리스트에서 원소를 하나씩 꺼내주는 일

```python
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 특성 -> 사이킷런이 기대하는 데이터 형태로 변환(2차원 리스트 or 리스트의 리스트)
fish_data = [[l, w] for l, w in zip(length, weight)] # 리스트 내포
fish_target = [1] * 35 + [0] * 14 # 정답 데이터
```

- k-최근접 알고리즘: 어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고 다술ㄹ 차지하는 것을 정답으로 사용 (주위 데이터로 현재 데이터 판단)

```python
# K-최근접 이웃 알고리즘 구현한 클래스인 KNeighborsClassfier 임포트!
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier() # 객체 생성
kn.fit(fish_data, fish_target) # 훈련: 특성과 정답 데이터를 전달하여 모델 훈련
kn.score(fish_data, fish_target) # 정확도: 훈련된 사이킷런 모델의 성능을 측정
kn.predict([[30, 600]]) # sample 넣을 때, 2차원 형태로 넣어줘야 함

# 산점도 그리기
import matplotlib.pyplot as plt # matplotlib의 pyplot 함수를 plt로 줄여서 사용
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length') # x축은 길이
plt.ylabel('weight') # y축은 무게
plt.show()
```

- 사실, k-최근접 이웃 알고리즘을 위해 준비해야 할 일은 데이터를 모두 가지고 있는 것이 전부 -> 그러므로 데이터가 아주 많은 경우 사용하기 어려움
	
	- 실제로 무언가 훈련되는 것이 없는 셈 

```python
# 클라스의 _fit_X 속성에 fish_data, _y 속성에 fish_target 값이 들어가 있음 
# print(kn._fit_X)
# print(kn._y)

# 이웃 개수를 49로 설정할 경우, 
kn49 = KNeighborsClassifier(n_neighbors=49) # 설정 안하면, 기본값 = 5
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
# 전체 샘플 크기(49)가 기준값이므로 무슨 값이든 도미라고 판단 -> 그러므로 정확도 71%
print(35/49) 
``` 
```python
## 정확도 100%가 깨지는 이웃 개수 찾기
for n in range(5, 50):
  # k-최근접 이웃 개수 설정
  kn.n_neighbors = n
  # 점수 계산
  score = kn.score(fish_data, fish_target)
  # 100% 정확도에 미치지 못하는 이웃 개수 출력
  if score < 1:
    print(n, score)
    break
```

---------------------------------------------------------------------------------

- 정리하기

	- 특성: 데이터를 표현하는 하나의 성질
	- 훈련: 머신러닝 알고리즘이 데이터에서 규칙을 찾는 과정 -> fit()
	- k-최근접 이웃 알고리즘: 어떤 규칙을 찾기보다 전체 데이터를 메모리에 가지고 있는 것이 전부
	- 모델: 알고리즘이 구현된 객체
	- 정확도: 정확한 답을 몇 개 맞혔는지 백분율 -> (정확히 맞힌 개수)/(전체 데이터 개수)

	- 핵심 패키지와 함수
		- scatter(): 산점도를 그리는 맷플롯립 함수
		- KNeighborsClassifier(): k-최근접 이웃 분류 모델을 만드는 사이킷런 클래스
		- fit(): 훈련/ 처음 두 매개변수로 훈련에 사용할 특성과 정답 데이터를 전달 
		- predict(): 예측/ 특성 데이터 하나만 매개변수로 받음
		- score(): 성능 측정/ 처음 두 매개변수로 특성과 정답 데이터를 전달
		
---------------------------------------------------------------------------------

- 전체코드

```python
# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

## 첫번째 머신러닝 프로그램

length = bream_length + smelt_length
weight = bream_weight + smelt_weight
# 특성 -> 사이킷런이 기대하는 데이터 형태로 변환(2차원 리스트 or 리스트의 리스트)
fish_data = [[l, w] for l, w in zip(length, weight)] # 리스트 내포
fish_target = [1] * 35 + [0] * 14 # 정답 데이터

# K-최근접 이웃 알고리즘 구현한 클래스인 KNeighborsClassfier 임포트!
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier() # 객체 생성
kn.fit(fish_data, fish_target) # 훈련: 특성과 정답 데이터를 전달하여 모델 훈련
kn.score(fish_data, fish_target) # 정확도: 훈련된 사이킷런 모델의 성능을 측정
kn.predict([[30, 600]]) # sample 넣을 때, 2차원 형태로 넣어줘야 함

# 산점도 그리기
import matplotlib.pyplot as plt # matplotlib의 pyplot 함수를 plt로 줄여서 사용
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length') # x축은 길이
plt.ylabel('weight') # y축은 무게
plt.show()

# 클라스의 _fit_X 속성에 fish_data, _y 속성에 fish_target 값이 들어가 있음 
# print(kn._fit_X)
# print(kn._y)

# 이웃 개수를 49로 설정할 경우, 
kn49 = KNeighborsClassifier(n_neighbors=49) # 설정 안하면, 기본값 = 5
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
# 전체 샘플 크기(49)가 기준값이므로 무슨 값이든 도미라고 판단 -> 그러므로 정확도 71%
print(35/49) 

## 정확도 100%가 깨지는 이웃 개수 찾기
for n in range(5, 50):
  # k-최근접 이웃 개수 설정
  kn.n_neighbors = n
  # 점수 계산
  score = kn.score(fish_data, fish_target)
  # 100% 정확도에 미치지 못하는 이웃 개수 출력
  if score < 1:
    print(n, score)
    break
```