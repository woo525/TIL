# CH3: 회귀 알고리즘과 모델 규제
---------------------------------------------------------------------------------

### 3-1: k-최근접 이웃 회귀

- 지도 학습의 한 종류인 회귀 문제를 이해하고 k-최근접 이웃 알고리즘을 사용해 농어의 무게를 예측하는 회귀 문제를 풀어보자!

- 지도 학습 알고리즘은 크게 '분류'와 '회귀'로 나뉨, 회귀: 임의의 어떤 숫자를 예측 
	- ex) 내년도 경제 성장률 예측, 배달이 도착하는 시간을 예측
	- 정해진 클래스가 없고 임의의 수치를 출력 
	- 회귀 -> 두 변수의 상관관계를 분석하는 방법

- k-최근접 이웃 회귀: 예측하려는 샘플에 가장 가까운 샘플 k개를 선택 -> 평균 

- 농어 데이터 준비: <http://bit.ly/perch_data>
- 여기서는 바로 넘파이 배열에서 만들기

```python
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 산점도 그리기
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트와 데이터 세트 분류
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 사이킷런에 사용할 훈련 세트는 2차원 배열!
# 그러므로 1차원 배열 -> 1개의 열이 있는 2차원 배열로 바꾸자!
# ex)
test_array = np.array([1,2,3,4])
#print(test_array.shape)
test_array = test_array.reshape(2, 2) # reshape(): 바꾸려는 배열의 크기 지정
#print(test_array.shape)

# 적용
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
#print(train_input.shape, test_input.shape) -> 2차원 배열로 성공적으로 변환

# k-최근접 이웃 알고리즘 훈련
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
# 회귀의 경우에는 정확한 숫자를 맞힌다는 것은 거의 불가능, 왜냐하면 예측하는 값이나 타깃 모두 임의의 수치이기 때문임
# 회귀의 경우에는 조금 다른 값으로 평가 -> 결정 계수(R^2)  

# 하지만 정확도처럼 R^2가 직감적으로 얼마나 좋은지 이해하기는 어려움

# mean_absolute_error은 타깃과 예측의 절댓값 오차를 평균하여 반환
from sklearn.metrics import mean_absolute_error
# 테스트 세트에 대한 예측
test_prediction = knr.predict(test_input)
# 테스트 세트에 대한 평균 절댓값 오차를 계산
mae = mean_absolute_error(test_target, test_prediction)
print(mae) # 이 정도의 오차가 나는 것

# 훈련 세트의 R^2(결정계수) 점수를 확인
print(knr.score(train_input, train_target))

# 과대적합: 테스트 세트 점수 < 훈련 세트 점수
# 과소적합: 테스트 세트 점수 > 훈련 세트 점수 or 둘다 낮음

# 과소적합이 일어나는 이유: 훈련 세트와 테스트 세트의 크기가 매우 작기 때문임

# **과소적합을 해결하기 위해 더 복잡한 모델 고안: k값을 5->3 ~ 국지적 패턴에 민감
# 여기서 반대로 이웃을 늘리면, 데이터 전반에 있는 일반적인 패턴을 따를 것임

knr.n_neighbors = 3 # 이웃의 개수를 3으로 설정
knr.fit(train_input, train_target) # 모델을 다시 훈련
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target)) # 과소적합 문제 해결


## 복잡한 모델과 단순한 모델 비교
#   k-최근접 이웃 회귀 객체 만들기
knr = KNeighborsRegressor()
# 5~45 x 좌표 만들기
x = np.arange(5, 45).reshape(-1, 1)

# n = 1, 5, 10일 때 예측 결과를 그래프로 그리기
for n in [1, 5, 10]:
  #모델 훈련
  knr.n_neighbors = n
  knr.fit(train_input, train_target)
  # 지정한 범위 x에 대한 예측을 구하기
  prediction = knr.predict(x)
  # 훈련 세트와 예측 결과를 그래프로 그리기 
  plt.scatter(train_input, train_target)
  plt.scatter(test_input, test_target)
  plt.plot(x, prediction)
  plt.show()
```

##### 정리하기

- 농어의 높이, 길이 등의 수치로 무게를 예측!
- 회귀: 임의의 수치를 예측하는 문제 -> 타깃값도 임의의 수치
- k-최근접 이웃 회귀 모델: 가까운 k개의 이웃을 찾고 이웃 샘플의 타깃값을 평균하여 해당 샘플의 예측값으로 사용
	- 회귀 모델의 점수로 R^2(회귀 모델의 성능 측정 도구 0에서 1), 즉 결정계수 값을 반환 -> 1에 가까울수록 좋음
	- 정량적인 평가를 하고 싶다면 다른 평가 도구 사용: 절댓값 오차 등
- 과대적합 -> 모델을 덜 복잡하게 만들어야 함 -> k값을 늘리기
- 과소적합 -> 모델을 더 복잡하게 만들어야 함 -> k값을 줄이기

- KNeighborsRegressor(): k-최근접 이웃 회귀 모델을 만드는 사이킷런 클래스
- mean_absolute_error(): 회귀 모델의 평균 절댓값 오차를 계산
- reshape(): 배열의 크기를 바꾸는 메서드

---------------------------------------------------------------------------------

### 3-2: 선형 회귀

- k-최근접 이웃 회귀와 선형 회귀 알고리즘의 차이를 이해하고 사이킷런을 사용해 여러 가지 선형 회귀 모델을 만들어보자!

- k-최근점 이웃 -> 새로운 샘플이 훈련 세트의 볌위를 벗어나면 엉뚱한 값을 예측하는 문제

- 농어 데이터 준비: <http://bit.ly/perch_data>

```python
import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꾸기
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=3) # 최근접 이웃 개수를 3으로 지정
# k-최근접 이웃 회귀 모델을 훈련
knr.fit(train_input, train_target)

# 50cm 농어의 무게 예측: 1033g BUT, 실제로 훨씬 더 무거움!
print(knr.predict([[50]]))

import matplotlib.pyplot as plt
# 50cm 농어의 이웃을 구하기
distances, indexes = knr.kneighbors([[50]])
# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그리기
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 50cm 농어 데이터 그리기
plt.scatter(50, 1033, marker='^')
plt.show()

# 이웃 샘플 타깃의 평균
print(np.mean(train_target[indexes]))
print(knr.predict([[100]])) # 길이가 100cm인 농어도 마찬가지로 1033g으로 예측!

# 한번 더 그래프를 그려 확인
# 100cm 농어의 이웃을 구함
distances, indexes = knr.kneighbors([[100]])
# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그리기
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 100cm 농어 데이터
plt.scatter(100, 1033, marker='^')
plt.show()


## 선형 회귀
# 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# 선형 회귀 모델을 훈련
lr.fit(train_input, train_target)
# 50cm 농어에 대해 예측
print(lr.predict([[50]]))

# y = ax + b 에서 a: coef_ b: intercept_
print(lr.coef_, lr.intercept_)

# 직선을 그려보자!
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프를 그리기
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50,1241.8, marker='^')
plt.show()

# 결정계수(R^2) 점수를 확인해보자!
print(lr.score(train_input, train_target)) # 훈련 세트
print(lr.score(test_input, test_target)) # 테스트 세트

# 왼쪽으로 직선이 나아가면 무게가 음수가 되는 구간 발생...
# + 둘다 점수가 낮은 과소적합!

# 직선을 곡선으로 만들자: 길이의 제곱을 입력데이터 앞에 붙여라!
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
print(train_poly.shape, test_poly.shape) # 열이 2개로 늘어남

# train_poly를 사용해 선형 회귀 모델 다시 훈련
# 타깃값은 그대로 사용 -> 목표하는 값은 어떤 그래프를 훈련하든지 바꿀 필요가 없음
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
# 모델이 훈련한 계수와 절편
print(lr.coef_, lr.intercept_)

# 무게를 왕길이와 길이의 선형관계로 표현!

## 훈련 세트의 산점도에 그래프로 그리기
# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만들기
point = np.arange(15, 50)
# 훈련 세트의 산점도를 그리기
plt.scatter(train_input, train_target)
# 15에서 49까지 2차 방정식 그래프를 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.show()

# 훈련 세트와 테스트 세트의 R^2 점수 평가
print(lr.score(train_poly, train_target)) # 상당히 개선되었음
print(lr.score(test_poly, test_target)) # BUT, 아직 과소적합 문제가 남아있음
```

##### 정리하기

- k-최근접 이웃 회귀를 사용해서 농어의 무게를 예측했을 때 발생하는 큰 문제는 훈련 세트 범위 밖의 샘플을 예측할 수 없다는 점 -> 이 문제를 해결하기 위해 선형 회귀를 사용!

- 선형 회귀는 훈련 세트에 잘 맞는 직선의 방정식을 찾는 것 = 최적의 기울기와 절편을 구한다는 것

- 농어의 무게가 음수가 되는 문제 발생 -> 다항 회귀 사용

- 선형 회귀는 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식, 특성이 하나면 직선 방정식
  - 특성과 타깃 사이의 관계는 선형 방정식의 계수 또는 가중치(기울기와 절편을 모두 의미)에 저장  

- 모델 파라미터: 가중치처럼 머신러닝 모델이 특성에서 학습한 파라미터
: 다항식을 사용하여 특성과 타깃 사이의 관계를 나타냄, 여전히 선형 회귀로 표현

- LinearRegression은 사이킷런의 선형 회귀 클래스
  - coef_: 특성에 대한 계수를 포함한 배열, 이 배열의 크기는 특성의 개수
  - intercept_: 절편

-----------------------------------------------------------------------------

### 3-3: 특성 공학과 규제

- 여러 특성을 사용한 다중 회귀에 대해 배우고 사이킷런의 여러 도구를 사용해 봅니다. 복잡한 모델의 과대적합을 막기 위한 릿지와 라쏘 회귀를 배웁니다.

- 다중회귀: 여러 개의 특성을 사용한 선형 회귀
- 특성 공학: 기존의 특성을 사용해서 새로운 특성을 뽑아내는 작업

- 판다스: 유명한 데이터 분석 라이브러리!
  - 데이터프레임: 판다스의 핵심 데이터 구조

- 데이터: <http://bit.ly/perch_data>

```python
import pandas as pd # pd는 관례적으로 사용하는 판다스의 별칭
df = pd.read_csv('http://bit.ly/perch_csv')
perch_full = df.to_numpy()
# print(perch_full)

import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트와 테스트 세트로 나누기
from sklearn .model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

# 변환기: fit, transform
# 추정기: fit, predict, score

# 우리가 사용할 변환기: PolynomialFeatures 클래스
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]])) # 특성이 많아짐!

# 사이킷런의 선형 모델은 자동으로 절편 추가
poly = PolynomialFeatures(include_bias=False) # 절편을 위한 항 제거
poly.fit([[2, 3]])
print(poly.transform([[2, 3]])) # 특성의 제곱과 특성끼리 곱한 항만 추가

# 이제 이 방식으로 train_input에 적용
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
poly.get_feature_names() # 9개의 특성이 각각 어떤 특성의 조합으로 만들어졌는지

# 마찬가지로 테스트 세트 변환
# **항상 훈련 세트 기준으로 테스트 세트 변환하는 습관 들이기** 
test_poly = poly.transform(test_input)


## 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

# 5제곱까지 특성을 만들어 출력
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape) # 특성의 개수가 무려 55개...

# 다시 훈련
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

### 규제!!
## 여기서. 과대적합을 줄이는 다른 방법을 배워보자! 특성 개수 줄이는 것 말고!
# 우선, 정규화를 먼저 시켜야 함: StandardScaler 클래스 -> 변환기의 일종
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly) 
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly) # 꼭 훈련세트로 학습한 변환기 사용!!

## 릿지 모델 훈련
# 릿지 회귀 -> 계수를 제곱한 값을 기준으로 규제!
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# alpha 매개변수: 릿지와 라쏘 모델을 사용할 때 규제의 강도를 조절
# 적절한 alpha 값을 찾는 한가지 방법: alpha 값에 대한 R^2 값의 그래프 그리기
# 점수가 가장 가까운 지점이 최적의 alpha 값
import matplotlib.pyplot as plt
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  # 릿지 모델 만들기
  ridge = Ridge(alpha=alpha)
  # 릿지 모델 훈련
  ridge.fit(train_scaled, train_target)
  # 훈련 점수와 테스트 점수를 저장
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

# 그래프로 그려보자!
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.show()

# 그러므로 alpha 값을 0.1로 하여 최종 모델을 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘 모델 훈련: 라쏘 회귀 -> 계수의 절댓값을 기준으로 규제!
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target)) # 라쏘도 과대적합을 잘 억제!

# 테스트 세트의 점수도 확인
print(lasso.score(test_scaled, test_target))

# 여기에서도 앞에서와같이 alpha 값을 바꾸어 가며 훈련 세트와 테스트 세트 점수 계산
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  # 라쏘 모델 만들기
  # 사이킷런의 라쏘 모델은 최적의 개수를 찾기 위해 반복적인 계산을 수행
  # 지정한 반복횟수가 부족할 때 경고 메시지 -> max_iter=10000 지정, BUT, 상관없다 
  lasso = Lasso(alpha=alpha, max_iter=10000)
  # 라쏘 모델 훈련
  lasso.fit(train_scaled, train_target)
  # 훈련 점수와 테스트 점수를 저장
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))
  
# 그래프로 그려보자!
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.show()

# 모델 훈련
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# 라쏘 모델은 계수 값을 아예 0으로 만들 수 있음
print(np.sum(lasso.coef_==0))
# 55-40=15 -> 55개의 특성을 모델에 주입했지만 라쏘 모델이 사용한 특성은 단, 15개
# 그러므로 라쏘 모델은 유용한 특성을 골라내는 용도로도 사용
```

##### 정리하기

- 모델의 과대적합 제어하기
  
  - 문제 상황: 선형 회귀 알고리즘 사용하여 모델 훈련 -> 과소적합 발생
  - 이에 다항 특성을 많이 추가 -> BUT, 반대로 과대적합
  - 이를 제약하기 위한 도구 필요

  - 릿지회귀: 선형 모델의 계수를 작게 만들어 과대적합을 완화, 비교적 효과가 좋아 널리 사용하는 규제 방법 -> 제곱 기준 
  - 라쏘회귀: 계수 값을 아예 0으로 만들 수도 있음 -> 절댓값 기준

- 다중 회귀: 여러 개의 특성을 사용하는 회귀 모델, 특성이 많으면 선형 모델은 강력한 성능 발휘

- 특성 공학: 주어진 특성을 조합하여 새로운 특성을 만드는 일련의 작업

- 하이퍼파라미터: 머신러닝 알고리즘이 학습하지 않는 파라미터, 사람이 사전에 지정해야 함

- pandas

  - read_csv(): CSV 파일을 로컬 컴퓨터나 인터넷에서 읽어 판다스 데이터프레임으로 변환하는 함수

- scikit-learn

  - PolynomialFeatures: 주어진 특성을 조합하여 새로운 특성 만듦
  - Ridge, Lasso

-----------------------------------------------------------------------------
