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

### 5-2: 교차 검증과 그리드 서치

- 검증 세트가 필요한 이유를 이해하고 교차 검증에 대해 배웁니다. 그리드 서치와 랜덤 서치를 이용해 최적의 성능을 내는 하이퍼파라미터를 찾아보자!

- 로지스틱 회귀 -> 특성공학, 규제, 가중치, 계수... 설명 어렵다!
- 결정트리 -> 가지치기 ex) max_depth... 설명 쉽다!

- 문제: 이런저런 값으로 모델을 많이 만들어서 테스트 세트로 평가하면 결국 테스트 세트에 잘 맞는 모델이 만들어지는 것 아닌가??
	
	- 검증 세트: 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련 세트를 또 나누는 것

```python
# 훈련 세트에서 모델을 훈련 -> 검증 세트로 모델을 평가 -> 매개변수를 바꿔가며 가장 좋은 모델 고름
# 훈련 세트와 검증 세트를 합쳐 전체 훈련 데이터에서 모델을 다시 훈련 -> 마지막에 테스트 세트에서 최종 점수를 평가

import pandas as pd
wine = pd.read_csv('http://bit.ly/wine-date')
#wine.head()

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# 다시 train_test_split() 함수에 넣어 훈련 세트와 검증 세트 만들기
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
# 훈련 세트와 검증 세트 크기 확인
print(sub_input.shape, val_input.shape)


## 모델을 만들고 평가
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))


# 교차 검증: 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터를 사용할 수 있음
# -> 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복

## 교차 검증: 5-폴드 교차 검증(기본값)
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores) # 여기서, test_score: 검증 폴드의 점수

# 분할기를 사용하여 교차 검증할 때 훈련 세트를 섞기: 회귀모델-KFold(), 분류모델-StratifiedKFold()
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
import numpy as np
print(np.mean(scores['test_score'])) # 검증 폴드 점수들의 평균

# 훈련 세트 섞은 후 10-폴드 교차 검증을 수행
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
import numpy as np
print(np.mean(scores['test_score'])) # 검증 폴드 점수들의 평균


## 하이퍼파라미터 튜닝: 그리드 서치 
# GridSearchCV 클래스는 친절하게도 하이퍼파라미터 탐색과 교차 검증을 한번에 수행

# 결정 트리 모델에서 min_impurity_decrease 매개변수의 최적값 찾기
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1) # n_jobs: 시스템에 있는 모든 코어 사용

# 그리드 서치 객체는 결정 트리 모델의 해당 파라미터를 바꿔가며 총 5번 실행
# 여기서, cv 기본값은 5 -> 5-폴드 교차 검증 -> 5*5=25개의 모델을 훈련!!

gs.fit(train_input, train_target)

# 훈련이 끝나면 25개의 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈련 세트에서 자동으로 다시 모델 훈련!

# 검증 점수가 가장 높은 모델은 gs.best_estimator_ 에 저장
dt = gs.best_estimator_
print(dt.score(train_input, train_target))

# 최적의 매개변수는 best_params_속성에 저장
print(gs.best_params_)

# 5번의 교차 검증으로 얻은 점수
print(gs.cv_results_['mean_test_score'])

best_index = np.argmax(gs.cv_results_['mean_test_score']) # 가장 큰 값의 인덱스를 추출
print(gs.cv_results_['params'][best_index])


## 더 복잡한 매개변수 조합 탐색
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
# 최상의 매개변수 확인
print(gs.best_params_)
# 최상의 교차 검증 점수 확인
print(np.max(gs.cv_results_['mean_test_score']))

# 하지만, 매개변수 간격을 0.0001 혹은 1로 설정함 -> 이렇게 간격을 둔 것에 특별한 근거 X
# 이보다 더 좁거나 넓은 간격으로 시도해 볼 수 있지 않을까??


## 랜덤 서치: 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달
# 싸이파이(수치 계산 전용 라이브러리)에서 2개의 확률 분포 클래스 임포트

from scipy.stats import uniform, randint
# randint는 정수값, uniform은 실수값 랜덤으로 뽑음
rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0, 1)
ugen.rvs(10)

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), 
                        params, n_iter=100, n_jobs=-1, random_state=42) # 총 100번을 샘플링 교차 검증
gs.fit(train_input, train_target)
# 그리드 서치보다 훨씬 교차 검증 수를 줄이면서 넓은 영역을 효과적으로 탐색
# 최적의 매개변수 조합 출력
print(gs.best_params_)
# 최고의 교차 검증 점수 확인
print(np.max(gs.cv_results_['mean_test_score']))
```

##### 정리하기

- 테스트 세트를 사용하면 결국 테스트 세트에 맞춰 모델을 훈련
	- 테스트 세트는 최종 모델을 훈련할 때까지 사용 X

- 검증 세트: 훈련 세트 중 일부를 다시 덜어 내어 만듦
	- 하이퍼파라미터 튜닝을 위해 모델을 평가할 때, 테스트 세트를 사용하지 않기 위함
	
- 교차 검증: 검증 세트를 한번 나누어 모델을 평가하는 것에 그치지 않고 여러 번 반복할 수 있음
	- 최종 검증 점수는 모든 폴드의 검증 점수 평균하여 계산

- 그리드 서치: 하이퍼파라미터 탐색을 자동화해 주는 도구 -> 마지막으로 최적의 매개변수 조합으로 최종 모델 훈련 
- 랜덤 서치: 연속된 매개변수 값을 탐색할 때 유용 -> 탐색 값을 샘플링할 수 있는 확률 분포 객체 전달 -> 지전된 횟수 만큼 샘플링

- scikit-learn

	- cross_validate(): 교차 검증을 수행하는 함수
	- GridSearchCV: 교차 검증으로 하이퍼파라미터 탐색을 수행 -> 최상의 모델을 찾은 후, 훈련 세트 전체를 사용해 최종 모델 훈련
	- RandomizedSearchCV: 교차 검증으로 랜덤한 하이퍼파라미터 탐색을 수행 -> 위와 동일

-----------------------------------------------------------------------------

### 5-3: 트리의 앙상블

- 앙상블 학습이 무엇인지 이해하고 다양한 앙상블 학습 알고리즘을 실습을 통해 배움

- 정형 데이터: ML, 특성공학 적용 O ex) 텍스트, 오디오, 이미지
- 비정형 데이터: DL, 특성공학 적용 X ex) DB, 엑셀, CSV

- 정형 데이터를 다루는데 가장 뛰어난 성과를 내는 알고리즘 -> 앙상블 학습(대부분 결정트리를 기반)
- 비정형 데이터 -> 신경망 알고리즘

```python
## 랜덤 포레스트
# 앙상블 학습의 대표 주자 중 하나로 안정적인 성능 덕분에  널리 사용
# 결정트리를 랜덤하게 만들어 결정트리의 숲 만듦 -> 각 결정트리의 예측을 사용해 최종 예측

# 부트스트랩: 데이터 세트에서 중복을 허용하여 데이터를 샘플링하는 방식(중복 -> 과대적합 방지) 
# 랜덤 포레스트 -> 랜덤하게 선택한 샘플과 특성을 사용하기 때문에 훈련 세트에 과대적합되는 것을 막아줌
# -> 검증 세트와 테스트 세트에서 안정적인 성능 

# RandomForestClassifier 클래스 -> 화이트 와인 분류 문제에 적용
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('http://bit.ly/wine-date')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# cross_validate() 함수를 사용해 교차 검증을 수행
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, 
                        return_train_score=True, n_jobs=-1)
# 훈련 세트와 검증 세트의 점수를 비교하면 과대적합을 파악하는데 용이
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 랜덤 포레스트 모델을 훈련 세트에 훈련한 후, 특성 중요도 출력
rf.fit(train_input, train_target)
print(rf.feature_importances_)

# oob_score=True로 지정하고 모델을 훈련하여 OOB 점수를 출력
# OOB 샘플: 부트스트랩 샘플에 포함되지 않고 남는 샘플 -> 이를 사용 -> 검증 세트 역할!
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)


## 엑스트라 트리
# 램덤 포레스트와의 차이점: 부트스트랩 샘플을 사용하지 않음
# 즉, 각 결정트리를 만들 때 전체 훈련 세트를 사용
# 대신 노드를 분할할 때 가장 좋은 분할을 찾는 것이 아니라 무작위로 분할
# 엑스트라 트리가 사용하는 결정 트리가 바로 splitter='random'인 결정 트리

# 엑스트라 트리 모델의 교차 검증 점수 확인!
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, 
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 보통 엑스트라 트리가 무작위성이 좀 더 크기 때문에 랜덤 포레스트보다 더 많은 결정 트리를 훈련
# BUT, 랜덤하게 노드를 분할하기 때문에 빠른 계산 속도가 엑스트라 트리의 장점

# 엑스트라 트리 모델 훈련 세트에 훈련한 후 특성 중요도 출력
et.fit(train_input, train_target)
print(et.feature_importances_)


## 그레이디언트 부스팅
# 깊이가 얕은 결정 트리를 사용하여 이진 트리의 오차를 보완하는 방식으로 앙상블 학습 하는 방법
# 기본적으로 깊이가 3인 결정 트리를 100개 사용 -> 과대적합에 강하고 일반적으로 높은 일반화 성능!!

# GradientBoostingClassifier -> 와인 데이터셋의 교차 검증 점수 확인
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, 
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 그레이디언트 부스팅은 결정 트리의 개수를 늘려도 과대적합에 매우 강함
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, 
                                random_state=42) # 경정 트리의 개수 500개!
scores = cross_validate(gb, train_input, train_target, 
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 특성 중요도
gb.fit(train_input, train_target)
print(gb.feature_importances_)


## 히스토그램 기반 그레이디언트 부스팅
# 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘
# 입력 특성을 256개의 구간으로 나눔 -> 노드를 분할할 때 최적의 분할을 매우 빠르게 찾을 수 있음

## 와인 데이터셋에 히스토그램 기반 그레이디언트 부스팅 적용
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, 
                        return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 과대적합을 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능을 제공

# 특성 중요도 -> 다양한 특성을 골고루 잘 평가
hgb.fit(train_input, train_target)
print(rf.feature_importances_)
# 테스트 세트에서의 성능 최종적으로 확인
hgb.score(test_input, test_target)


## XGBoost를 사용해 와인 데이터의 교차 검증 점수를 확인
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, 
                        return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

## LightGBM를 사용해 와인 데이터의 교차 검증 점수를 확인
from lighgtgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, 
                        return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

##### 정리하기

- 앙상블 학습을 통한 머신러닝 성능 향상!
	
	- 앙상블 학습: 더 좋은 예측 결과를 만들기 위해 여러 개의 결정 트리 모델을 훈련하는 머신러닝 알고리즘

	- 랜덤 포레스트: 결정 트리를 훈련하기 위해 부트스트랩을 샘플을 만들고 전체 특성 중 일부를 랜덤하게 선택하여 결정 트리 만듦
		- 결정 트리 기반의 앙상블 학습 방법, 부트스트랩 샘플을 사용 + 랜덤하게 일부 특성을 선택

	- 엑스트라 트리: 부트스트랩 샘플을 사용하지 않고 노드를 분할할 때 최선이 아니라 랜덤하게 분할 -> 훈련 속도가 빠르지만 보통 더 많은 트리 필요
		- 랜덤하게 노드를 분할해 과대적합을 감소시킴

	- 그레이디언트 부스팅: 깊이가 얕은 트리를 연속적으로 추가하여 손실 함수를 최소화하는 앙상블 방법
		- 랜덤 포레스트나 엑스트라 트리와 달리 결정 트리를 연속적으로 추가하여 손실 함수를 최소화
		- 훈련 속도가 조금 느리지만 좋은 성능

	- 히스토그램 기반 그레이디언트 부스팅: 훈련 데이터를 256개의 구간으로 변환하여 사용 -> 노드 분할 속도가 매우 빠름

- scikit-learn

	- RandomForestClassifier: 랜덤 포레스트 분류 클래스
	- ExtraTreesClassifier: 엑스트라 트리 분류 클래스
	- GradientBoostingClassifier: 그레이디언트 부스팅 분류 클래스
	- HistGradientBoostingClassifier: 히스토그램 기반 그레이디언트 부스팅 분류 클래스
	
-----------------------------------------------------------------------------








