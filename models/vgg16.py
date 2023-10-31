import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16


### Parameters

# Image Params
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

# Grid Params
GRID_ROWS = 6
GRID_COLS = 8

# Model Params
IMG_PATH = "../images"
RESIZED_WIDTH = IMG_WIDTH
RESIZED_HEIGHT = IMG_HEIGHT
LABELS = GRID_ROWS * GRID_COLS
BATCH_N = 32
EPOCHS = 50
# TRAIN_STEPS = int(len(train_data)/BATCH_N)
VAL_STEPS = 6
DENSE_UNITS = 1024



### Data Generation
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2, # validation용으로 사용할 데이터의 비율
        rescale=1./255,       # 0 ~ 255 를 0 ~ 1 로 변경
        brightness_range=(0.2, 1.0)   # Range for picking a brightness shift value from. 0 == black, 1 == normal
        )

train_data_gen = image_gen.flow_from_directory(
                      directory=IMG_PATH,         # 디렉터리 지정
                      batch_size=BATCH_N,         # 한번에 생성할 데이터의 크기 설정
                      # target_size=(RESIZED_WIDTH, RESIZED_HEIGHT),  # 변경될 이미지 데이터의 크기
                      classes = [i for i in range(LABELS)],           # 클래스 번호 부여: 디렉터리 순
                      class_mode = 'sparse',      # Label을 숫자로 표기
                      subset='training'           # Training 용 데이터: 전체의 80%
                      )

test_data_gen = image_gen.flow_from_directory(
                      directory=IMG_PATH,         # 디렉터리 지정
                      batch_size=BATCH_N,         # 한번에 생성할 데이터의 크기 설정
                      # target_size=(RESIZED_WIDTH, RESIZED_HEIGHT),  # 변경될 이미지 데이터의 크기
                      classes = [i for i in range(LABELS)],           # 클래스 번호 부여: 디렉터리 순
                      class_mode = 'sparse',      # Label을 숫자로 표기
                      subset='validation'         # Testing 용 데이터: 전체의 20%
                      )


### Load VGG16 Model
model = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    )
# model.summary()


### Set trainable option
for layer in model.layers[:-4]:
    layer.trainable = False
for layer in model.layers:
    print(layer, layer.trainable)


### Layer를 추가하여 모델 완성
model_fine = tf.keras.models.Sequential()
model_fine.add(model)
model_fine.add(Flatten())
model_fine.add(Dense(DENSE_UNITS, activation='relu'))
model_fine.add(BatchNormalization())
model_fine.add(Dense(LABELS, activation='softmax'))
# model_fine.summary()


### Compile Model
model_fine.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


### Training
# %%time
history = model_fine.fit(
                train_data_gen,
                # steps_per_epoch=TRAIN_STEPS,
                epochs=EPOCHS,
                validation_data=test_data_gen,
                validation_steps=VAL_STEPS,
                callbacks = [cp_callback]
                )


### Ploting
loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title('Accuray')
plt.plot(epochs, history.history['accuracy'], 'r', label='accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'g', label='val_accuracy')
plt.grid(True)
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.title('Loss')
plt.plot(epochs, history.history['loss'], 'r', label='loss')
plt.plot(epochs, history.history['val_loss'], 'g', label='val_loss')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()


### 결과 출력을 위한 함수
def Make_Result_Plot(suptitle, data, label, y_max):
    fig_result, ax_result = plt.subplots(2, 5, figsize=(18, 7))
    fig_result.suptitle(suptitle)
    for idx in range(10):
        ax_result[idx//5][idx%5].imshow(data[idx],cmap="binary")
        ax_result[idx//5][idx%5].set_title("test_data[{}] (label : {} / y : {})".format(idx, label[idx], y_max[idx]))

### 학습 후 상황
y_out = model_fine.predict(test_data)
y_max = np.argmax(y_out, axis=1).reshape((-1, 1))
Make_Result_Plot("After Training", test_data, test_labels, y_max)