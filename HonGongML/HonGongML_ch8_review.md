# CH8: 이미지를 위한 인공 신경망
-----------------------------------------------------------------------------

### 8-1: 합성곱 신경망의 구성 요소

- 합성곱 신경망을 구성하는 기본 개념과 동작 원리를 배우고 간단한 합성곱, 풀링 계산 방법을 익힙니다.

##### 합성곱: 입력 데이터에 마법의 도장을 찍어서 유용한 특성만 드러나게 하는 것

	- 입력 데이터 전체에 가중치를 적용하는 것이 아니라 일부에 가중치를 곱함
	- 첫번째 합성곱에 사용된 가중치 w1~w3과 절편 b가 두번째 합성곱에도 동일하게 사용 

	- 장점: 2차원 입력에도 적용 가능 -> 필터(도장)도 2차원!! 여기서, 커널 크기는 '하이퍼 파라미터' 
	- 합성곱 계산을 통해 얻은 출력 -> '특성맵'

	- 합성곱 층에서도 여러 개의 필터를 사용 -> 만들어진 특성 맵은 순서대로 차곡차곡 쌓임
	- 2차원 구조를 그대로 사용하기 때문에 합성곱 신경망이 이미지 처리 분야에서 뛰어난 성능

##### 케라스의 합성곱 층

```python
from tensorflow import keras
```

	- 합성곱은 Conv2D 클래스로 제공
	- 매개변수: 필터(도장)의 개수, 필터에 사용할 커널의 크기, 활성화 함수
```python 
keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu')
```

	- 특성맵은 활성화 함수를 이미 통과한 값!
	- 합성곱 신경망: 1개 이상의 합성곱 층을 쓴 인공 신경망

##### 패딩과 스트라이드

	- 커널의 크기는 (3, 3)으로 그대로 두고 출력의 크기를 입력과 동일하게 (4, 4)로 만들려면??
	- (4, 4) 입력과 동일한 크기의 출력을 만들려면 마치 더 큰 입력에 합성곱하는 척해야 함 -> '패딩'

	- 패딩: 입력 배열 주위를 가상의 원소로 채우는 것
	- 커널이 도장(필터)을 찍을 횟수를 늘려주기 위해서 입력 배열 주변을 가상의 원소로 채우는 것

	- 세임 패딩: 입력 주위에 0으로 패딩하는 것 -> 합성곱 신경망에서는 세임 패딩이 많이 사용
	- 밸리드 패딩: 패딩 없이 순수한 입력 배열에서만 합성곱을 하여 특성맵을 만드는 경우
		- 밸리드 패딩은 특성 맵의 크기가 줄어들 수 밖에 없음

	- 그럼 왜 합성곱에서는 패딩을 즐겨 사용할까요??
		- 입력을 이미지라고 생각하면 *모서리에 있는 중요한 정보가 특성 맵으로 잘 전달되지 않을 가능성*이 높음
		- 즉, 패딩을 하지 않을 경우 중앙부와 모서리 픽셀이 합성곱에 참여하는 비율은 크게 차이가 남

	- 적절한 패딩 -> 이미지의 주변에 있는 정보를 잃어버리지 않도록 도와줌
	- 케라스 Conv2D 클래스에서는 padding 매개변수로 패딩을 지정할 수 있음
```python
keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu', padding='same')
```

	- 지금까지 본 합성곱 연산은 좌우, 위아래로 한 칸씩 이동 -> 두 칸씩 건너뛸 수도 있음
	- 이런 이동의 크기 -> '스트라이드'
```python
keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu', padding='same', strides=1)
```

	- strides 매개변수: 오른쪽으로 이동하는 크기와 아래쪽으로 이동하는 크기를 (1, 1)과 같이
	- 튜플을 사용해 각각 지정할 수 있음 -> 대부분 기본값을 그대로 사용!!

	- 세임 패딩의 경우, 입력과 만들어진 특성맵의 가로세로 크기가 같다는 점

##### 풀링

	- 합성곱 층에서 만든 특성맵의 가로세로 크기를 줄이는 역할 수행
	- 특성맵에 커널없는 필터를 적용하는 것과 같음 -> 가중치가 없음, 가장 큰 값이나 평균값 계산
	- 풀링층의 출력도 '특성맵'

	- 풀링에서는 겹치지 않고 이동
	- 케라스에서는 MaxPooling2D 클래스로 풀링을 수행
```python
keras.layers.MaxPooling2D(2) # 풀링의 크기: 2
keras.layers.MaxPooling2D(2, strides=2, padding='valid') # 위의 코드와 동일
```

	- 평균 풀링을 제공하는 클래스: AveragePooling2D
	- 많은 경우 평균 풀링보다 최대 풀링을 많이 사용
	- 평균 풀링은 특성 맵에 있는 중요한 정보를 (평균하여) 희석시킬 수 있기 때문

	- 합성곱 신경망: 합성곱 층에서 특성맵 생성 -> 풀링에서 크기를 줄이는 구조

##### 컬러 이미지를 사용한 합성곱

	- 컬러 이미지는 RGB(빨강, 초록, 파랑) 채널로 구성되어 있기 때문에 컴퓨터는 이를 3차원 배열로 표시

	- 하나의 컬러 이미지는 너비와 높이 차원 외에 깊이 차원(또는 채널 차원)이 있음
	- 그러므로 필터(도장)도 '깊이'가 필요!

	- 입력이나 필터의 차원이 몇 개인지 상관없이 항상 출력은 하나의 값
	- 특성 맵에 있는 한 원소가 채워짐 + 케라스의 합성곱 층은 3차원 입력을 기대

	- 합성곱 신경망에서 '필터'는 이미지에 있는 어떤 특징을 찾는다고 생각할 수 있음
	- 층이 깊어질수록 다양하고 구체적인 특징을 감지 -> 필터의 개수 늘리기
	- 어떤 특징이 이미지의 어느 위치에 놓이더라도 쉽게 감지 -> 너비와 높이 차원 줄이기

	- 합성곱 층과 풀링 층은 거의 항상 함께 사용
	- 최대 풀링을 즐겨 사용하며 특성 맵을 절반으로 줄임
	- 마지막에는 특성맵을 1차원 배열로 펼쳐서 1개 이상의 밀집층에 통과시켜 클래스에 대한 확률 만듦

##### 정리하기

- 합성곱: 입력과 가중치를 곱하고 절편을 더하는 선형 계산
	- 각 합성곱은 입력 전체가 아니라 일부만 사용하여 선형 계산 수행
	- 필터: 밀집층의 뉴런에 해당, 필터의 가중치와 절편을 종종 '커널'이라 부름

- 특성맵: 합성곱 층이나 풀링 층의 출력 배열을 의미

- 패딩: 합성곱 층의 입력 주위에 추가한 0으로 채워진 픽셀 
	- ex) 세임 패딩, 밸리드 패딩

- 스트라이드: 합성곱 층에서 필터가 입력 위를 이동하는 크기

- 풀링: 가중치가 없고 특성 맵의 가로세로 크기를 줄이는 역할 수행  

-----------------------------------------------------------------------------

### 8-2: 합성곱 신경망을 사용한 이미지 분류

- 케라스 API를 사용해 합성곱 신경망 모델을 만들어 패션 MNIST 이미지를 분류하는 방법을 배웁니다.
- 텐서플로를 사용하면 사용자는 직관적으로 신경망을 설계할 수 있습니다! 

```python
## 패션 MNIST 데이터 불러오기
# 합성곱 신경망은 2차원 이미지를 그대로 사용하기 때문에 이렇게 일렬로 펼치지 않음
# 입력 이미지는 항상 깊이(채널) 차원이 있어야 함 -> Conv2D 층을 사용하기 위해 마지막에 이 채널 추가!
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# train_scaled 의 차원 -> (50000, 28, 28, 1)


## 합성곱 신경망 만들기
# 합성곱 신경망의 구조: 합성곱 층으로 이미지에서 특징을 감지한 후, 밀집층으로 클래스에 따른 분류 확률 계산!
model = keras.Sequential()
# 32개의 필터, 커널의 크기: (3,3), 렐루 활성화 함수, 세임 페딩
# 케라스 신경망 모델의 첫번째 층에서 입력의 차원을 지정해 주어야 함
# 첫번째 합성곱-풀링층 -> (3, 3, 1)필터의 개수 32개
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', 
                              padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2)) # 풀링층 추가

# 세임 패딩을 적용했기 때문에 합성곱 층에서 출력된 특성 맵의 가로세로 크기는 입력과 동일
# 그다음 (2,2) 풀링을 적용 -> 특성맵의 크기 절반
# 합성곱 층에서 32개의 필터를 사용 -> 특성맵의 깊이는 32 -> (14, 14, 32)

# 두번째 합성곱-풀링층 -> (3, 3, 32)필터의 개수 64개
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', 
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2)) # 풀링층 추가 -> (7, 7, 64)

# 3차원 특성맵을 일렬로 펼칠 차례 -> 마지막에 10개의 뉴런을 가진 (밀집) 출력층에서 확률을 계산하기 때문
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4)) # 은닉층의 과대적합을 막아 성능을 조금 더 개선
model.add(keras.layers.Dense(10, activation='softmax'))
# 클래스 10개를 분류하는 다중 분류 문제이므로 마지막 층의 활성화 함수는 소프트맥스를 사용
# 이렇게 합성곱 신경망의 구성을 마침
# summary() 메서드로 모델 구조를 출력
model.summary()
# 층의 구성을 그림으로 표현해주는 plot_model() 함수
keras.utils.plot_model(model)
# 입력과 출력의 크기 표시
keras.utils.plot_model(model, show_shapes=True, 
                       to_file='cnn-architecture.png', dpi=300)


## 패션 MNIST 데이터에 적용할 합성곱 신경망 모델의 구성을 마침 -> 이제 모델을 컴파일하고 훈련
# Adam 옵티마이저를 사용하고 ModelCheckpoint 콜백과 EarlyStopping 콜백을 함께 사용해 조기 종료 기법을 구현
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, 
                                                  restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, 
                    validation_data=(val_scaled, val_target), 
                    callbacks=[checkpoint_cb, early_stopping_cb])

# 얼핏 보아도 훈련 세트의 정확도가 이전보다 훨씬 좋아짐
# 손실 그래프를 그려서 조기 종료가 잘 이루어졌는지 확인
import matplotlib.pyplot as plt 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# EarlyStopping 클래스에서 restore_best_weights 매개변수를 True로 지정했으므로
# 현재 model 객체가 최적의 모델 파라미터로 복원
# 세트에 대한 성능 평가
model.evaluate(val_scaled, val_target)
# fit() 메서드의 출력 중 여덟 번째 에포크의 출력과 동일

# predict() 메서드 -> 새로운 데이터에 대해 예측
# 맷플롯립에서는 흑백 이미지에 깊이 차원은 없음
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

# 10개의 클래스에 대한 예측 확률 출력
preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', 
           '가방', '앵클 부츠']
# preds 배열에서 가장 큰 인덱스를 찾아 classes 리스트의 인덱스로 사용
import numpy as np
print(calsses[np.argmax(preds)])

# 합성곱 신경망을 만들고 훈련하여 새로운 샘플에 대한 예측을 수행하는 방법도 알아봄
# 마지막으로 맨 처음에 떼어 놓았던 테스트 세트로 합성곱 신경망의 일반화 성능을 가늠
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

# evaluate() 메서드로 테스트 세트에 대한 성능을 측정
model.evaluate(test_scaled, test_target)
# 91%의 성능을 기대할 수 있다!!
```

##### 정리하기

- 케라스의 Conv2D 클래스를 사용해 32개의 필터와 64개의 필터를 둔 2개의 합성곱 층을 추가했음
- 합성곱 신경망은 이미지를 주로 다루기 때문에 각 층의 출력을 시각화하기 좋음

- TensorFlow
	- Conv2D: 입력의 너비와 높이 방향의 합성곱 연산을 구현한 클래스
	- MaxPooling2D: 입력의 너비와 높이를 줄이는 풀링 연산을 구현한 클래스
	- plot_model(): 케라스 모델 구조를 주피터 노트북에 그리거나 파일로 저장

- matplotlib
	- bar(): 막대그래프를 출력

-----------------------------------------------------------------------------

### 8-3:

