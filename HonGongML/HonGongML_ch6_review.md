# CH6: 비지도 학습
-----------------------------------------------------------------------------

### 6-1: 군집 알고리즘

- 흑백 사진을 분류하기 위해 여러 가지 아이디어를 내면서 비지도 학습과 군집 알고리즘에 대해 이해!
- 비지도 학습: 타깃값이 없을 때, 데이터에 있는 패턴을 찾거나 데이터 구조를 파악하는 머신러닝 방식

```python
# 과일 사진 데이터 준비하기
!wget https://bit.ly/fruits_300 -O fruits_300.npy

import numpy as np
import matplotlib.pyplot as plt
fruits = np.load('fruits_300.npy') # 넘파이 배열 형태로 데이터 로드
print(fruits.shape) # 배열의 크기 -> 샘플의 개수 x 이미지 높이 x 이미지 너비

# 첫번째 행에 있는 픽셀 100개에 들어있는 값을 출력
print(fruits[0, 0, :])

# 흑백 이미지는 사진으로 찍은 이미지를 넘파이 배열로 변환할 때 반전 시킨 것
# 우리의 관심 대상은 바탕이 아니라 사과 -> 컴퓨터는 255에 가까운 값에 주목

# 첫번째 이미지를 그림으로 그려서 위 숫자와 비교
plt.imshow(fruits[0], cmap='gray')
plt.show()

# 배경이 검은색? 흑백 이미지를 반전하여 넘파이 배열에 저장!
plt.imshow(fruits[0], cmap='gray_r') # 우리가 보기 편하게 반전하여 이미지 출력
plt.show()

# 바나나와 파인애플 이미지도 출력
fig, axs = plt.subplots(1, 2) # 그래프를 쌓을 행과 열 지정: 1개의 행과 2개의 열
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()


## fruits 데이터를 사과, 파인애플, 바나나로 각각 나누어 보기
# 100 x 100 = 10000, 1차원 배열로~ 계산 편리!
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

# 배열의 크기 확인
print(apple.shape)

# 배열에 들어있는 샘플의 픽셀 평균값 계산
print(apple.mean(axis=1)) # 사과 샘플 100개에 대한 픽셀 평균값 계산

# 히스토그램 그려보기
plt.hist(np.mean(apple, axis=1), alpha=0.8) # axis는 기준축(방향으로 계산)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8) # alpha는 투명도
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# 픽셀 10000개에 대한 평균값을 막대그래프
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

# 픽셀 평균값을 100 x 100 크기로 바꿔서 이미지처럼 출력
# 모든 사진을 합쳐놓은 대표 이미지 역할
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()


## 평균값과 가까운 사진 고르기 - 사과
# 절댓값 오차 구하기
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

# 이 값이 가장 작은 순서대로 100개를 골라보자.
# apple_mean과 오차가 가장 작은 샘플 100개를 고르는 셈 
apple_index = np.argsort(abs_mean)[:100]
# np.argsort(): 작은 것에서 큰 순서대로 나열한 abs_mean 배열의 인덱스를 반환

fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()


## 평균값과 가까운 사진 고르기 - 바나나
# 절댓값 오차 구하기
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)
# 이 값이 가장 작은 순서대로 100개를 골라보자.
banana_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[banana_index[i*10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()
```

-----------------------------------------------------------------------------

### 6-2: k-평균

- k-평균 알고리즘의 작동 방식을 이해하고 과일 사진을 자동으로 모으는 비지도 학습 모델 만들어보자!

- k-평균 알고리즘의 작동 방식
	1. 무작위로 k개의 클러스터 중심을 정함
	2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
	3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경
	4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복

- k-평균 알고리즘은 처음에는 램덤하게 클러스터 중심을 선택하고 점차 가장 가까운 샘플의 중심으로 이동하는 비교적 간단한 알고리즘  

```python
!wget https://bit.ly/fruits_300 -O fruits_300.npy # 데이터 다운로드

import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# k-평균 알고리즘 학습
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

# 군집된 결과는 KMeans 클래스 객체의 labels_ 속성에 저장
print(km.labels_)
# 레이블 0, 1 ,2로 모은 샘플의 개수 확인
print(np.unique(km.labels_, return_counts=True))

# 그림으로 출력하기 위해 함수 만들기
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
  n = len(arr) # n은 샘플 개수
  # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수 계산
  rows = int(np.ceil(n/10))
  # 행이 1개이면 열의 개수는 샘플 개수 그렇지 않으면 10개
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols, 
                          figsize=(cols*ratio, rows*ratio), squeeze=False)
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n: # n개까지만 그리기
        axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
      axs[i, j].axis('off')
  plt.show()

# 불리언 인덱싱
draw_fruits(fruits[km.labels_==0]) # 사과
draw_fruits(fruits[km.labels_==1]) # 바나나
draw_fruits(fruits[km.labels_==2]) # 파인애플

# 클러스터의 중심, 이 배열을 이미지로 출력
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# 인덱스가 100인 샘플에서 각 클러스터 중심까지 거리
print(km.transform(fruits_2d[100:101]))

# 가장 가까운 클러스터 중심을 예측 클래스로 출력
print(km.predict(fruits_2d[100:101]))
draw_fruits(fruits[100:101])

print(km.n_iter_) # 알고리즘 반복 횟수


## 최적의 k값 찾기
# 엘보우 방법: 최적의 클러스터 개수 = 클러스터 개수와 이너셔 값의 그래프가 꺾이는 지점
# 이너셔: 클러스터 중심과 클러스터에 속한 샘플 사이의 거리 제곱합
inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.show()
```

-----------------------------------------------------------------------------

### 6-3: 주성분 분석

- 차원 축소에 대해 이해하고 대표적인 차원 축소 알고리즘 중 하나인 PCA(주성분 분석) 모델을 만들어보자!

- 차원 축소를 사용하면 데이터셋의 크기를 줄일 수 있고 비교적 시각화하기 쉬움

```python
!wget https://bit.ly/fruits_300 -O fruits_300.npy
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# PCA 클래스로 주성분 분석
from sklearn.decomposition import PCA
pca = PCA(n_components=50) # 주성분의 개수 지정
pca.fit(fruits_2d) # 비지도 학습이기 때문에 타깃값 제공 X

print(pca.components_.shape) # 확인

# 주성분을 100 X 100 크기의 이미지처럼 출력
import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
  n = len(arr) # n은 샘플 개수
  # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수 계산
  rows = int(np.ceil(n/10))
  # 행이 1개이면 열의 개수는 샘플 개수 그렇지 않으면 10개
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols, 
                          figsize=(cols*ratio, rows*ratio), squeeze=False)
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n: # n개까지만 그리기
        axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
      axs[i, j].axis('off')
  plt.show()

draw_fruits(pca.components_.reshape(-1, 100, 100))

# 원본 데이터를 주성분에 투영하여 특성의 개수를 10000개에서 50개로 줄이기
# 위에서 50개의 주성분을 찾은 PCA 모델을 사용 -> (300, 50) 크기의 배열로 변환
print(fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# 원본 데이터 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

# 이 데이터를 100 X 100 크기로 바꾸어 100개씩 나누어 출력
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
  draw_fruits(fruits_reconstruct[start:start+100])
  print("\n")


## 설명된 분산
print(np.sum(pca.explained_variance_ratio_))

# 설명된 분산의 비율을 그래프로 그리기 -> 적절한 주성분의 개수 찾기
plt.plot(pca.explained_variance_ratio_) # 처음 10개의 주성분이 대부분의 분산을 표현

# 로지스틱 회귀 모델을 사용하여 원본 데이터와 PCA로 축소된 데이터 지도 학습에 어떤 차이 있는지
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# 지도 학습이므로 타깃 데이터 만들기
target = np. array([0]*100 + [1]*100 + [2]*100)
# 원본 데이터 -> 교차 검증 -> 성능 분석
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
# PCA로 축소된 데이터로 실시
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 설명된 분산의 50%에 달하는 주성분을 찾도록 PCA 모델 만들기
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
# 몇개의 주성분을 찾았는지 확인
print(pca.n_components_)
# 이 모델로 원본 데이터를 변환
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape) # 주성분이 2개이므로 변환된 데이터의 크기는 (300, 2)

# 2개의 특성만 사용하고도 교차 검증의 결과가 좋은지 확인
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 차원 축소된 데이터를 사용해 k-평균 알고리즘으로 클러스터 찾기
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
# fruits_pca로 찾은 클러스터는 각각 91개, 99개, 110개의 샘플을 포함
# 2절에서 원본 데이터를 사용했을 때와 거의 비슷한 결과

# KMeans가 찾은 레이블을 사용해 과일 이미지 출력
for label in range(0, 3):
  draw_fruits(fruits[km.labels_ == label])
  print("\n")

# fruits_pca 데이터는 2개의 특성이 있기 때문에 2차원으로 표현
# 화면에 출력하기 비교적 쉽다!
for label in range(0, 3):
  data = fruits_pca[km.labels_ == label]
  plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
```

-----------------------------------------------------------------------------