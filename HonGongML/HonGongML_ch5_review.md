# CH5: 트리 알고리즘
-----------------------------------------------------------------------------

### 5-1: 결정 트리

- 결정 트리 알고리즘을 사용해 새로운 분류 문제를 다루자!

```python
## 로지스틱 회귀로 와인 분류하기
# 판다스를 사용해 인터넷에서 직접 데이터셋 불러오기
import pandas as pd
wine = pd.read_csv('http://bit.ly/wine-date')
# wine.head() # 제대로 불러왔는지 처음 5개의 데이터 확인

# wine.info() # 데이터프레임의 각 열의 데이터 타입과 누락 여부 확인
# wine.describe() # 열에 대한 간략한 통계를 출력: 최소, 최대, 평균값 등

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine[['class']].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42) # 20% 정도만 테스트 세트로!
# print(train_input.shape, test_input.shape) # 훈련 세트 5197개, 테스트 세트 1300개

# 훈련 세트 전처리 -> 같은 객체를 그대로 사용해 테스트 세트 변환
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target)) # 점수가 높지 않다!
# 과소적합 -> 변수 C를 바꿔볼까? solver 매개변수 다른 알고리즘 선택? 다양한 특성?

# 로지스틱 회귀 모델을 설명하기 위해 학습한 계수와 절편 출력
print(lr.coef_, lr.intercept_) # 계수와 절편을 가지고 모델의 작동 원리 설명하긴 어려움!


## 결정 트리 모델 훈련 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42) # 최적의 분할을 찾기 전에 특성의 순서를 섞음
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 훈련 세트
print(dt.score(test_scaled, test_target)) # 테스트 세트

# 결정 트리를 이해하기 쉬운 트리 그림으로 출력
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7)) # 그래프 사이즈
plot_tree(dt)
plt.show()

# 이를 좀 더 알아보기 쉽게~
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
# 결정 트리에서 예측하는 방법 -> 리프 노드에서 가장 많은 클래스가 예측 클래스
# 회귀일 경우, 리프 노드에 도달한 샘플의 타깃을 평균하여 예측값으로 사용

# 루트 노드는 어떻게 당도 -0.239를 기준으로 왼쪽과 오른쪽 노드로 나누었을까요?
# criterian 매개변수에 지정한 불순도를 사용
# 지니 불순도: 클래스의 비율을 제곱해서 더한 다음 1에서 빼면 됨

# *결정 트리 모델은 부모 노드와 자식 노드의 불순도 차이(정보 이득)가 가능한 크도록 트리를 성장시킴
# 정보 이득이 최대가 되도록 데이터를 나눔(노드를 분할) -> 지니 불순도 기준, 다른 불순도 변경 가능

# 앞의 트리는 제한 없이 자라났기 때문에 훈련 세트보다 테스트 세트 점수가 크게 낮음


## 가지치기: 과대적합 방지, 자라날 수 있는 트리의 최대 깊이를 지정
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

# 트리 그래프로 그리기
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 특성값의 스케일은 결정 트리 알고리즘에 아무런 영향을 미치지 않음 -> 표준화 전처리 X
# 전처리하기 전의 훈련 세트와 테스트 세트로 결정 트리 모델을 다시 훈련 -> 변함 X
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target)) 

# 그래프로 그리기 -> 특성값을 표준 점수로 바꾸지 않은 터라 이해하기 훨씬 쉽다!!
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 특성 중요도 출력
print(dt.feature_importances_)
```

##### 정리하기 

- 결정 트리

	- 특성을 추가하지 않고도 결정 트리의 성능이 로지스틱 회귀 모델보다 좋았음
	- 비교적 비전문가에게도 설명하기 쉬운 모델을 만듦
	- 많은 앙상블 학습 알고리즘의 기반 

	- 앙상블 학습 -> 신경망과 함께 가장 높은 성능을 내기 때문에 인기가 높은 알고리즘

- 결정 트리: 예/ 아니오에 대한 질문을 이어나가면서 정답을 찾아 학습하는 알고리즘
- 불순도: 최적의 질문을 찾기 위한 기준 -> 지니 불순도, 엔트로피 불순도
- 정보 이득: 부모 노드와 자식 노드의 불순도 차이 -> 최대화되도록 학습
- 가지치기: 결정 트리는 제한 없이 성장하면 훈련 세트에 과대적합되기 쉬움 -> 결정 트리의 성장을 제한하는 방법(규제)
- 특성 중요도: 결정 트리에 사용된 특성이 불순도를 감소하는데 기여한 정도를 나타내는 값

- pandas

	- info(): 데이터프레임의 요약된 정보를 출력
	- describe: 데이터프레임 열의 통계값을 제공

- scikit-learn

	- DecisionTreeClassifier: 결정 트리 분류 클래스
	- plot_tree(): 결정 트리 모델을 시각화  

-----------------------------------------------------------------------------

### 5-2: 