# CH4: 다양한 분류 알고리즘
-----------------------------------------------------------------------------

### 4-1: 로지스틱 회귀

- 로지스틱 회귀 알고리즘을 배우고 이진 분류 문제에서 클래스 확률을 예측

```python
# 데이터 준비
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv')
# fish.head() # 처음 5개 행 출력

# 어떤 종류의 생선이 있는지 Species 열에서 고유한 값 추출
print(pd.unique(fish['Species']))

## Species 열을 타깃으로 만들고 나머지 5개 열을 입력 데이터로 사용
# to_numpy() 배열을 이용하여 넘파이 배열로 바꿔주기
# 열을 선택하는 방법: 데이터 프레임에서 원하는 열을 리스트로 나열
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
# print(fish_input[:5])
fish_target = fish['Species'].to_numpy()

# 데이터를 훈련 세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 훈련 세트와 테스트 세트를 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기로 테스트 세트에 들어있는 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# 타깃 데이터에 2개 이상의 클래스가 포함된 문제 -> '다중분류'

# 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 '알파벳 순'
# 데이터 프레임에 저장되어 있는 순서와 다르다!
print(kn.classes_)

# 테스트 세트에 있는 처음 5개 샘플의 타깃값을 예측
print(kn.predict(test_scaled[:5]))

# 테스트 세트에 있는 처음 5개의 샘플에 대한 확률 출력
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4)) # 소수점 넷째 자리까지]
# predict_proba() 메서드의 출력 순서는 앞서 보았떤 class_ 속성과 동일

# kneighbors() 메서드의 입력은 2차원 배열이어야 함
# 이를 위해 넘파이 배열의 슬라이싱 연산자 사용
# **슬라이싱 연산자는 하나의 샘플만 선택해도 항상 2차원 배열 만들어짐**
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

## BUT, 문제 발생: 가능한 확률이 0/3, 1/3, 2/3, 3/3이 전부!

## 로지스틱 회귀: 이름은 회귀이지만 '분류' 모델!
# 시그모이드 함수(로지스틱 함수) 그리기
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.show()


## 로지스틱 회귀에서 이진 분류 수행하기

# 불리언 인덱싱: 넘파이 배열은 True, False 값을 전달하여 행을 선택  
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

# 도미와 빙어의 행만 골라내기
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 위 데이터로 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 처음 5개 샘플 예측
print(lr.predict(train_bream_smelt[:5])) 

# 예측 확률 출력
print(lr.predict_proba(train_bream_smelt[:5]))
# 첫번째 열이 음성 클래스(0)에 대한 확률, 두번째 열이 양성 클래스(1)
print(lr.classes_) # 빙어(Smelt)가 양성 클래스!

# 로지스틱 회귀로 성공적인 이진 분류 수행!

# 로지스틱 회귀가 학습한 계수를 확인
print(lr.coef_, lr.intercept_)

# z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# z값을 시그모이드 함수에 대입 <- 이중 분류
from scipy.special import expit
print(expit(decisions)) # predict_proba() 메서드의 두번째 열의 값과 동일


## 로지스틱 회귀로 '다중 분류' 문제 수행
lr = LogisticRegression(C=20, max_iter=1000) 
# max_iter 기본값은 100 # 규제 제어 매개변수: C -> 작을수록 규제 커짐, 기본값 1
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 테스트 세트의 처음 5개 샘플에 대한 예측
print(lr.predict(test_scaled[:5]))

# 테스트 세트의 처음 5개 샘플에 대한 예측 확률 -> 7개의 열
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
print(lr.classes_)

# 이진 분류는 샘플마다 2개의 확률을 출력
# 다중 분류는 샘플마다 클래스 개수만큼 확률을 출력 -> 이 중에서 가장 높은 확률이 예측 클래스

# 다중 분류는 클래스마다 z값을 하나씩 계산
print(lr.coef_.shape, lr.intercept_)
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 소프트맥스 함수(정규화된 지수 함수) 이용 예측 확률 계산 <- 다중 분류
from scipy.special import softmax
proba = softmax(decision, axis=1) # axis=1 각 행에 대한 소프트맥스 계산
print(np.round(proba, decimals=3))
```

##### 정리하기

- k-최근접 이웃 모델은 이웃한 샘플의 클래스 비율이므로 항상 정해진 확률만 출력
- 이 문제를 해결하기 위해 '로지스틱 회귀'를 사용

	- 로지스틱 회귀는 '분류' 모델이다!! BUT, 회귀처럼 선형 방정식 사용
	- 계산한 값을 그대로 출력하는 것이 아니라 위 값을 0-1 사이로 압축!

	- 이진 분류 -> 음성 클래스의 확률은 1에서 양성 클래스의 확률을 뺀다.
	- 다중 분류 -> 클래스 개수만큼 방정식을 훈련, 각 방정식의 출력값을 소프트 맥스 함수를 통과시켜 전체 클래스에 대한 합이 항상 1

	- *다음 절에서는 또 다른 머신러닝 알고리즘인 확률적 경사 하강법 배워보자!*

- 로지스틱 회귀: 선형 방정식을 사용한 분류 알고리즘, 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 사용하여 클래스 확률을 출력

- 다중 분류: 타깃 클래스가 2개 이상인 분류 문제, 로지스틱 회귀는 다중 분류를 위해 소프트맥스 함수를 사용하여 클래스를 예측

- 시그모이드 함수: 선형 방정식의 출력을 0과 1 사이의 값으로 압축하며 이진 분류를 위해 사용

- 소프트맥스 함수: 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화하여 합이 1이 되도록 만듦

- scikit-learn

	- LogisticRegression: 선형 분류 알고리즘인 로지스틱 회귀를 위한 클래스
	- predict_proba(): 예측 확률을 반환
	- decision_function(): 모델이 학습한 선형 방정식의 출력을 반환

-----------------------------------------------------------------------------

### 4-2: 확률적 경사 하강법

- 경사 하강법 알고리즘을 이해하고 대량의 데이터에서 분류 모델을 훈련하는 방법을 배우자!

- 확률적 경사 하강법: 대표적인 점진적 학습 알고리즘, 훈련 세트에서 랜덤하게 하나의 샘플을 골라 가장 가파른 경사를 따라 최적의 장소로 조금씩 이동하는 알고리즘 -> 실패하면 처음부터 다시!
	- 에포크: 확률적 경사 하강법에서 훈련 세트를 한 번 모두 사용하는 과정
	- 일반적으로 경사 하강법은 수십, 수백 번 이상 에포크를 수행
	
	- 미니배치 경사 하강법: 여러 개의 샘플을 사용해 경사 하강법을 수행하는 방식
	- 배치 경사 하강법: 극단적으로 한 번 경사로를 따라 이동하기 위해 전체 샘플을 사용

- 확률적 경사 하강법을 사용하는 알고리즘 -> 신경망 알고리즘

- 손실 함수: 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준 -> 값이 작을수록 좋음
	- BUT, 어떤 값이 최솟값인지 모름 -> 확률적 경사 하강법
	- 정확도는 듬성듬성하므로 손실 함수로 사용할 수 없음 -> 손실 한수는 미분 가능해야 함

	- 이진 분류: 로지스틱 손실 함수
	- 다중 분류: 크로스엔트로피 손실 함수

```python
## '확률적 경사 하강법'을 사용한 분류 모델을 만들어보자!
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv')
# print(fish.head()) # 판다스 데이터프레임 만들기

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy() # 입력 데이터
fish_target = fish[['Species']].to_numpy() # 타깃 데이터

# 훈련 세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 훈련 세트와 테스트 세트의 특성을 표준화 전처리
# 꼭 훈련 세트에서 학습한 통계 값으로 테스트 세트도 변환
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 사이킷런에서 확률적 경사 하강법을 제공하는 대표적인 분류용 클래스 임포트
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
# 1 에포크 이어서 다시 학습 -> partial_fit() 메서드 호출
sc.partial_fit(train_scaled, train_target)  #
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# partial_fit() 메서드만 사용하려면 훈련 세트에 있는 전체 클래스의 레이블을
# partial_fit() 메서드에 전달해주어야 함: up.unique() 함수로 train_target에 있는
# 7개의 생선 목록 만들기
import numpy as np
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

# 300번의 에포크 동안 훈련을 반복하여 진행
for _ in range(0, 300):
  sc.partial_fit(train_scaled, train_target, classes=classes)
  train_score.append(sc.score(train_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_target))

# 그래프로 그리기 -> 백 번째 에포크가 적절한 반복 횟수로 보임
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.show()

# SGDClassifier의 반복횟수를 100에 맞추고 모델을 다시 훈련
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 힌지 손실을 사용해 같은 반복 횟수 동안 모델을 훈련
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```

##### 정리하기

- 이 절에서 확률적 경사 하강법을 사용해 점진적으로 학습하는 로지스틱 회귀 모델 훈련

- 확률적 경사 하강법: 손실 함수라는 산을 정의하고 가장 가파른 경사를 따라 조금씩 내려오는 알고리즘
	- 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘

- 손실 함수: 확률적 경사 하강법이 최적화할 대상
	- 이진 분류: 로지스틱 회귀
	- 다중 분류: 크로스엔트로피 손실 함수
	- 회귀 문제: 평균 제곱 오차 손실 함수

- 에포크: 확률적 경사 하강법에서 전체 샘플을 모두 사용하는 한 번 반복

- 데이터가 매우 크기 때문에 데이터를 조금씩 사용해 점진적으로 학습하는 방법이 필요

- scikit-learn
	- SGDClassifier: 확률적 경사 하강법을 사용한 분류 모델 만듦
	- SGDRegressor: 확률적 경사 하강법을 사용한 회귀 모델 만듦
	 