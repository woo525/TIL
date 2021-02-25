# CH7: 비지도 학습
-----------------------------------------------------------------------------

### 7-1: 인공 신경망

- 딥러닝과 인공 신경망 알고리즘을 이해하고 텐서플로를 사용해 간단한 인공 신경망 모델을 만들어 봅니다. 

```python
# 텐서플로의 케라스 패키지를 임포트하고 패션 MNIST 데이터를 다운로드
from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()

# 전달받은 데이터의 크기 확인
print(train_input.shape, train_target.shape)
# 훈련 데이터: 60000개의 이미지, 크기는 28 x 28
# 타깃 데이터: 60000개의 원소가 있는 1차원 배열

# 테스트 세트의 크기도 확인
print(test_input.shape, test_target.shape)
# 테스트 세트: 10000개의 이미지

# 6장에서 맷플롯립 라이브러리로 과일을 출력했던 것처럼 
# 훈련 데이터에서 몇 개의 샘플을 그림으로 출력
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
  axs[i].imshow(train_input[i], cmap='gray_r')
  axs[i].axis('off') # 좌표축을 그리지 않음
plt.show() 

# 이 샘플들의 타깃값을 확인
# 파이썬의 '리스트 내포'를 사용해서 
# 처음 10개 샘플의 타깃값을 리스트로 만든 후 출력
print([train_target[i] for i in range(10)])

# 패션 MNIST의 타깃은 '0~9'까지의 숫자 레이블로 구성
# 넘파이 unique() 함수로 레이블 당 샘플 개수 확인
import numpy as np
print(np.unique(train_target, return_counts=True))
# 0~9까지 레이블마다 정확히 6000개의 샘플이 들어있음


## 로지스틱 회귀로 패션 아이템 분류하기
# 훈련 샘플이 60000개나 되기 때문에 샘플을 하나씩 꺼내서 모델을 훈련하는 방법이 더 효율적
# '확률적 경사 하강법': 여러 특성 중 기울기가 가장 가파른 방향을 따라 이동
# 이미지의 경우 보통 255로 나누어 0~1 사이의 값으로 정규화
# SGDClassifier는 2차원 입력을 다루지 못함 -> 1차원 배열로!
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
# reshape()메서드: 첫번째 차원(샘플 개수)은 변하지 않고
# 원본 데이터의 두번째, 세번째 차원이 1차원으로 합쳐짐
print(train_scaled.shape)
# 784개의 픽셀로 이루어진 60000개의 샘플이 준비됨
# SGDClassifier 클래스와 cross_validate 함수를 사용
# 교차 검증으로 성능을 확인
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=5, random_state=42) # 반복 횟수 5번
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
# 만족할 만한 수준 X


## 인공 신경망으로 모델 만들기
# 인공 신경망에서는 교차 검증을 잘 사용하지 않고 검증 세트를 별도로 덜어내어 사용
# 딥러닝 분야의 데이터셋은 충분히 큼 -> 검증 점수 안정적, 교차 검증 -> 시간 너무 많이 걸림
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
# 훈련 세트에서 20%를 검증 세트로 덜어냄
print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# 훈련 세트로 모델을 만들고 검증 세트로 평가
# 10개의 패션 아이템을 분류하기 위해 10개의 뉴런으로 구성
# 케라스의 레이어 패키지 안에는 다양한 층이 준비되어 있음 ex) 밀집층(완전 연결층)
# 케라스의 Dense 클래스를 사용해 '밀집층' 만들기
# 매개변수: 뉴런 개수, 뉴런의 출력에 적용할 함수, 입력의 크기
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))

# 신경망 층을 만들었다! 이제 이 밀집층을 가진 신경망 모델을 만들자!
model = keras.Sequential(dense) # 신경망 모델!

# 활성화 함수: 뉴런의 선형 방정식 결과에 적용되는 함수 ex) 소프트맥스 함수


## 인공 신경망으로 패션 아이템 분류하기
# 케라스 모델은 훈련하기 전에 '설정' 단계가 있음 -> model 객체의 compile() 메서드에서 수행
# 꼭 지정해야할 것은 손실 함수의 종류 + 훈련 과정에서 계산하고 싶은 측정값을 지정
# 다중 분류: 크로스 엔트로피 손실 함수 사용 -> loss='categorical_corssentropy'
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 다중 분류에서 크로스 엔트로피 손실 함수를 사용하려면 0, 1, 2와 같이 정수로 된 타깃값을 원-핫 인코딩으로 변환

# 패션 MNST 데이터의 타깃값은 어떻게 되어 있었나요?
print(train_target[:10]) # 모두 정수로 되어 있음
# 정수로 된 타깃값을 사용해 크로스 엔트로피 손실을 계산하는 것이 바로
# 'sparse_categorical_crossentropy' !!

# 케라스는 모델이 훈련할 때 기본으로 에포크마다 손실 값을 출력
# 정확도를 함께 출력하기 위해 metrics='accuracy' !!

# 훈련
model.fit(train_scaled, train_target, epochs=5) # 반복할 에포크 횟수 5번
# 에포크마다 걸린 시간과 손실, 정확도 출력
# 그럼 이제 따로 떼어 놓은 검증 세트에서 모델의 성능을 확인
# 케라스에서 모델의 성능을 평가하는 메서드 -> evaluate()
model.evaluate(val_scaled, val_target)
```

##### 정리하기

1. 사이킷런 모델
```python
sc=SGDClassifier(loss='log', max_iter=5) # 모델
sc.fit(train_scaled, train_target) # 훈련
sc.core(val_scaled, val_target) # 평가
```

2. 케라스 모델
```python
# 모델
dense=keras.layers.Dense(10, activation='softmax', input_shape(784,)) # 층 생성
model=keras.Sequential(dense)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
# 훈련
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
# 평가
```

- 인공 신경망: 생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘 (딥러닝)
- 텐서플로: 구글이 만든 딥러닝 라이브러리로 매우 인기 있음
- 밀집층(완전 연결층): 가장 간단한 인공 신경망의 층, 특별히 출력층에 밀집층을 사용할 때는 분류하려는 클래스와 동일한 개수의 뉴런을 사용
- 원-핫 인코딩: 정숫값 배열에서 해당 정수 위치의 원소만 1이고 나머지 모두 0으로 반환, 다중 분류에서 출력층에서 만든 확률과 크로스 엔트로피 손실을 계산하기 위함

-----------------------------------------------------------------------------

### 7-2: 심층 신경망

- 인공 신경망에 층을 여러 개 추가하여 패션 MNIST 데이터셋을 분류하면서 케라스로 심층 신경망을 만드는 방법을 자세히 배웁니다. 

```python
## 2개의 층

# 케라스 API를 사용해 패션 MNIST 데이터셋 불러오기
from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()

# 이미지의 픽셀값을 0~255 범위에서 0~1 사이로 변환
# 28*28 크기의 2차원 배열을 784 크기의 1차원 배열로 펼치기
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 인공 신경망 모델에 층을 2개 추가!!
# 입력층과 출력층 사이에 밀집층이 추가 -> '은닉층'
# 활성화 함수: 신경망 층의 선형 방정식의 계산 값에 적용하는 함수

# 출력층에 적용하는 활성화 함수는 종류가 제한 -> 시그모이드, 소프트맥스
# 은닉층의 활성화 함수는 비교적 자유로움 -> 시그모이드, 렐루
# '회귀'일 경우는 사용할 필요가 없음 -> 출력이 임의의 어떤 숫자이기 때문

# 시그모이드 활성화 함수를 사용한 '은닉층'과 소프트맥스 함수를 사용한 '출력층' 만들기
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')


## 심층 신경망 만들기
model = keras.Sequential([dense1, dense2])
# *주의* : '출력층'을 가장 마지막에 두어야 한다!

# 모데의 summary() 메서드를 호출하면 층에 대한 정보 얻을 수 있음
model.summary()

# 케라스 모델의 fit() 메서드에 훈련 데이터를 주입하면 이 데이터를 한 번에 모두
# 사용하지 않고 잘게 나누어 여러 번에 걸쳐 경사 하강법 단계를 수행 -> 미니배치 경사 하강법
# 케라스의 기본 미니배치 크기: 32개

# 샘플마다 784개의 픽셀값이 은닉층을 통과하면서 100개의 특성으로 압축


## 층을 추가하는 다른 방법
# Sequential 클래스의 생성자 안에서 바로 Dense 클래스의 객체를 만드는 경우가 많음
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,),
                       name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
    ], name='패션 MNIST 모델')

model.summary()

# Sequential 클래스에서 층을 추가할 때 *가장 널리 사용하는 방법*은 모델의 add() 메서드
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()


## 이제 모델을 훈련해보자!
# compile() 메서드의 설정은 1절에서 했던 것과 동일
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# 렐루 함수: 입력이 양수일 경우 마치 활성화 함수가 없는 것처럼 그냥 입력을 통과
# 음수일 경우 0으로 만듦 -> max(0, z) 
# *이미지 처리에서 좋은 성능*

# Flatten 층: 입력층 바로 뒤에 오며, 배치 차원 제외 나머지 입력 차원을 모두 일렬로 펼치는 역할
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) 
# Flatten 클래스는 학습하는 층이 아니므로 깊이에 포함 X -> 깊이 2 
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary() # Flatten 층을 신경망 모델에 추가하면 입력값의 차원을 짐작 가능

# 이렇게 입력 데이터에 대한 전처리 과정을 가능한 모델에 포함시키는 것이 케라스 API 철학 중 하나!!

# 그럼 훈련 데이터를 다시 준비해서 모델을 훈련
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 모델을 컴파일하고 훈련하는 것은 다음 코드처럼 이전과 동일
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
# 시그모이드 함수를 사용했을 때와 비교하면 성능이 조금 향상

# 검증 세트에서의 성능도 확인!
model.evaluate(val_scaled, val_target)
# 1절의 은닉층을 추가하지 않은 경우보다 몇 퍼센트 성능이 향상


## 인공 신경망의 하이퍼파라미터에 대해 잠시 알아보자!
# 여러가지 옵티마이저를 테스트하자!
# 가장 기본적인 옵티마이저는 '확률적 경사 하강법 SGD'
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')
# 다음 코드와 동일
# sgd = keras.optimizers.SGD()
# model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', 
#               metrics='accuracy')

# 만약 클래스의 학습률을 바꾸고 싶다면
# sgd = keras.optimizers.SGD(learning_rate=0.1)
# sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)

# 적응적 학습률: 모델이 최적점에 가까이 갈수록 학습률을 낮출 수 있음
# 안정적으로 최적점에 수렴 + 학습률 매개변수를 튜닝하는 수고를 덜 수 있음
# 적응적 학습률을 사용하는 대표적 옵팀마이저: Adagrad, RMSprop
# adagrad = keras.optimizers.Adagrad()
# model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', 
#               metrics='accuracy')
# rmsprop = keras.optimizers.RMSprop()
# model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', 
#               metrics='accuracy')

# Adam: 모멘텀 최적화와 RMSprop의 장점을 접목
# 적응적 학습률 사용 위 3개의 클래스는 learning_rate 매개변수의 기본값 모두 0.001

## Adam 클래스의 매개변수 기본값을 사용해 패션 MNIST 모델 훈련!!
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) 
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
```

##### 정리하기

- 심층 신경망: 2개 이상의 층을 포함한 신경망
- 렐루 함수: 이미지 분류 모델의 은닉층에 많이 사용하는 활성화 함수, 시그모이드 함수는 층이 많을수록 활성화 함수의 양쪽 끝에서 변화가 작기 때문에 학습이 어려워짐
- 옵티마이저: *신경망의 가중치와 절편을 학습하기 위한 알고리즘 또는 방법*, 케라스에는 다양한 경사 하강법 알고리즘이 구현되어 있음 ex) SGD, 네스테로프 모멘텀, RMSprop, Adam 등이 있음

-----------------------------------------------------------------------------

### 7-3: 신경망 모델 훈련

- 인공 신경망 모델을 훈련하는 모범 사례와 필요한 도구들을 살펴보겠습니다. 이런 도구들을 다뤄 보면서 텐서플로와 케라스 API에 더 익숙해 질 것입니다.

```python
# 케라스의 fit() 메서드는 History 클래스 객체를 반환
# History 객체에는 훈련 과정에서 계산한 지표, 즉 손실과 정확도 값이 저장

# 이 값을 사용하면 그래프를 그릴 수 있음

# 패션 MNIST 데이터셋을 적재 + 훈련 세트와 검증 세트로 나눔
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 모델을 만드는 간단한 함수 정의
def model_fn(a_layer=None):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  model.add(keras.layers.Dense(100, activation='relu'))
  if a_layer:
    model.add(a_layer)
  model.add(keras.layers.Dense(10, activation='softmax'))
  return model

# 단순하게 model_fn() 함수 호출
model = model_fn()
model.summary()

# fit() 메서드의 결과를 history 변수에 담아보자!
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
# verbose 매개변수: 0-훈련 과정 나타내지 않음, 1-진행 막대+손실 등의 지표, 2-막대X

# history 객체 -> 훈련 측정값이 담겨 있는 history 딕셔너리가 들어 있음
print(history.history.keys()) # 손실과 정확도가 포함되어 있음
# 에포크마다 계산한 값이 순서대로 나열된 단순한 리스트


# 맷플롯립을 사용해 쉽게 그래프로 그리자!
import matplotlib.pyplot as plt

# 손실 출력
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 정확도 출력
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 확실히 에포크마다 손실이 감소하고 정확도가 향상
# 에포크 횟수를 20으로 늘려서 모델을 훈련하고 손실 그래프 그리자!
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
# 예상대로 손실이 잘 감소!!

# 이전보다 더 나은 모델을 훈련한 것일까? 이전에 배웠던 것 중에 놓친 것이 있지 않을까?

# 4장에서는 정확도를 사용하여 과대/과소적합을 설명 -> 이 장에서는 손실을 사용하여 과대/과소적합을 다뤄보자!
# *인공 신경망 모델이 최적화하는 대상은 정확도가 아니라 '손실 함수'*

# 에포크마다 검증 손실을 계산하기 위해 케라스 모델의 fit() 메서드에 검증 데이터 전달 
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))
# 검증 세트에 대한 손실은 'val_loss'에 들어 있고 정확도는 'val_accuracy'에 들어있음
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'legend'])
plt.show()

# 검증 손실이 상승하는 시점을 가능한 뒤로 늦추면 검증 세트에 대한 손실이 줄어들
# 뿐만 아니라 검증 세트에 대한 정확도도 증가

# 옵티마이저 하이퍼파라미터를 조정하여 과대적합을 완화
# Adam은 적응적 학습률 사용 -> 에포크가 진행되면서 학습률의 크기를 조정
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'legend'])
plt.show()
# 과대적합이 훨씬 줄어듦

# 더 나은 손실 곡선을 얻으려면 학습률을 조정해서 다시 시도


## 신경망에서 사용하는 대표적인 규제 방법에 대해 알아보자!

## 드롭아웃
# 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서 과대적합을 막음
# 이전 층의 일부 뉴런이 랜덤하게 꺼지면 특정 뉴런에 과대하게 의존하는 것을 줄임
# 모든 입력에 주의를 기울이게 됨

# 앙상블하는 것처럼 상상할 수 있음 -> 앙상블은 과대적합을 막아 주는 아주 좋은 기법

# model_fn() 함수에 드롭아웃 객체를 전달하여 30% 정도 드롭아웃 해보자!
model = model_fn(keras.layers.Dropout(0.3))
model.summary()

# 훈련이 끝난 뒤에 평가나 예측을 수행할 때는 드롭아웃을 적용하지 말아야 함
# *텐서플로와 케라스는 모델을 평가와 예측에 사용할 때는 자동으로 드롭아웃을 적용 X

# 훈련 손실과 검증 손실의 그래프를 그려 비교
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'legend'])
plt.show() # 과대 적합이 확실히 줄었음

## 모델 저장과 복원
# 에포크 횟수를 10으로 다시 지정하고 모델을 훈련
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=10, verbose=0, 
                    validation_data=(val_scaled, val_target))

# 훈련된 모델의 파라미터를 저장하는 save_weights()
model.save_weights('model-weights.h5')
# 모델 구조와 모델 파라미터를 함께 저장하는 save()
model.save('model-whole.h5')

# 이 두 파일이 잘 만들어졌는지 확인
!ls -al *.h5

# 훈련을 하지 않은 새로운 모델 -> model-weight.h5 파일에서 훈련된 모델 파라미터 읽어서 사용
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')
# 검증 정확도 확인
# 케라스에서 예측을 수행하는 predict() 메서드: 샘플마다 10개의 클래스에 대한 확률을 반환
# 10개 확률 중에 가장 큰 값을 골라 타깃 레이블과 비교하여 정확도를 계산
import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
# argmax(): 배열에서 가장 큰 값의 인덱스 반환, axis=-1: 배열의 마지막 차원을 따라 최댓값
print(np.mean(val_labels == val_target)) # 정확도!!

# model-whole.h5 파일에서 새로운 모델을 만들어 바로 사용
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)
# 위와 같은 모델을 저장하고 다시 불러들였기 때문에 동일한 정확도


## 콜백
# 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체
# ModelCheckpoint 콜백: 기본적으로 최상의 검증 점수를 만드는 모델을 저장
# 저장될 파일 이름을 'best-model.h5'로 지정하여 콜백 적용
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
model.fit(train_scaled, train_target, epochs=20, verbose=0, 
          validation_data=(val_scaled, val_target), 
          callbacks=[checkpoint_cb])
# 모델이 훈련한 후에 best-model.h5에 최상의 검증 점수를 낸 모델이 저장
model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target)


# 조기종료: 과대적합이 시작되기 전에 훈련을 미리 중지, 딥러닝 분야에서 널리 사용
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, 
                                                  restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
          validation_data=(val_scaled, val_target), 
          callbacks=[checkpoint_cb, early_stopping_cb])
# 훈련을 마치고 나면 몇 번째 에포크에서 훈련이 중지되었는지 early_stopping_cb 객체의 stopped_epoch 속성에서 확인
print(early_stopping_cb.stopped_epoch) 

# 훈련 손실과 검증 손실 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

# 조기 종료로 얻은 모델을 사용해 검증 세트에 대한 성능을 확인
model.evaluate(val_scaled, val_target)

# 컴퓨터 자원과 시간을 아낄 수 있고 ModelCheckpoint 콜백과 함께 사용하면 최상의 모델을 자동으로 저장
```

##### 정리하기

- fit() 메서드의 반환값을 사용해 훈련 세트와 검증 세트에 대한 손실을 그래프로 그릴 수 있음 -> fit() 메서드는 훈련 세트뿐만 아니라 검증 세트를 전달할 수 있는 매개변수를 제공

- 신경망에서 즐겨 사용하는 대표적인 규제 방법인 '드롭아웃': 일부 뉴런의 출력을 랜덤하게 꺼서 일부 뉴런에 의존하는 것을 막고 마치 많은 신경망을 앙상블 하는 효과 -> 은닉층에 있는 뉴런의 출력을 랜덤하게 꺼서 과대적합을 막는 기법

- 콜백: 케라스 모델을 훈련하는 도중에 어떤 작업을 수행할 수 있도록 도와주는 도구 -> 대표적으로 최상의 모델을 자동으로 저장해 주거나 검증 점수가 더 이상 향상되지 않은면 조기 종료

- 조기 종료: 검증 점수가 더 이상 감소하지 않고 상승하여 과대적합이 일어나면 훈련을 계속 진행하지 않고 멈추는 기법

-----------------------------------------------------------------------------

   