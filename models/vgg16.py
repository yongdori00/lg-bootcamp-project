import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16


### Parameters

# Data Params
IMG_WIDTH = 212
IMG_HEIGHT = 320
IMG_CHANNELS = 3
IMG_PATH = "./cropped"
CSV_PATH = "./output.csv"
DF_XCOL = "name"
DF_YCOL = ["row", "column"]

# Grid Params
GRID_ROWS = 6
GRID_COLS = 8

# Model Params
RESIZED_WIDTH = IMG_WIDTH
RESIZED_HEIGHT = IMG_HEIGHT
LABELS = GRID_ROWS * GRID_COLS
BATCH_N = 32
EPOCHS = 3
# TRAIN_STEPS = int(len(train_data)/BATCH_N)
VAL_STEPS = 6
DENSE_UNITS = 1024


### Dataframe 불러오기
df = pd.read_csv(CSV_PATH)


### Data Generation
image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2, # validation용으로 사용할 데이터의 비율
        rescale=1./255,       # 0 ~ 255 를 0 ~ 1 로 변경
        brightness_range=(0.2, 1.0)   # Range for picking a brightness shift value from. 0 == black, 1 == normal
        )

train_data_gen = image_gen.flow_from_dataframe(
                      dataframe=df,         # 디렉터리 지정
                      directory=IMG_PATH,
                      x_col=DF_XCOL,
                      y_col=DF_YCOL,
                      batch_size=BATCH_N,         # 한번에 생성할 데이터의 크기 설정
                      target_size=(RESIZED_WIDTH, RESIZED_HEIGHT),                     # 변경될 이미지 데이터의 크기
                      # classes = [(i//GRID_COLS, i%GRID_COLS) for i in range(LABELS)],  # 클래스 번호 부여: 디렉터리 순
                      class_mode = 'multi_output', # Label을 숫자로 표기
                      subset='training'           # Training 용 데이터: 전체의 80%
                      )

test_data_gen = image_gen.flow_from_dataframe(
                      dataframe=df,         # 디렉터리 지정
                      directory=IMG_PATH,
                      x_col=DF_XCOL,
                      y_col=DF_YCOL,
                      batch_size=BATCH_N,         # 한번에 생성할 데이터의 크기 설정
                      target_size=(RESIZED_WIDTH, RESIZED_HEIGHT),                     # 변경될 이미지 데이터의 크기
                      # classes = [(i//GRID_COLS, i%GRID_COLS) for i in range(LABELS)],  # 클래스 번호 부여: 디렉터리 순
                      class_mode = 'multi_output', # Label을 숫자로 표기
                      subset='validation'         # Testing 용 데이터: 전체의 20%
                      )

print("******************************")
train_data_t, train_labels_t = next(train_data_gen)
print(np.array(train_data_t).shape)
print(np.array(train_labels_t).shape)
print(train_labels_t)

### Load VGG16 Model
model = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, IMG_CHANNELS))
    )
print('original VGG16')
# print(model.summary())


### Set trainable option
for layer in model.layers[:-4]:
    layer.trainable = False
# for layer in model.layers:
#     print(layer, layer.trainable)


### Layer 추가
x = Flatten()(model.layers[-1].output)
x = Dense(DENSE_UNITS, activation='relu')(x)
x = BatchNormalization()(x)

# 첫번째 출력: ROW
row_out = Dense(GRID_ROWS, activation='softmax', name='row_out')(x)

# 두번째 출력: COL
col_out = Dense(GRID_COLS, activation='softmax', name='col_out')(x)

# 최종 모델
model_out = tf.keras.models.Model(inputs=model.input, outputs=[row_out, col_out])
print('fine tuned model')
print(model_out.summary())


### Loss function
# loss_weights = {'row_out': 1.0, 'col_out': 1.0}  # 각 출력에 대한 가중치
# loss = {'row_out': 'categorical_crossentropy', 'col_out': 'categorical_crossentropy'}  # 각 출력에 대한 손실 함수


### Model compile
model_out.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        loss_weights=1.0,
        metrics=['accuracy'])


### Training
# %%time
history = model_out.fit(
                train_data_gen,
                # steps_per_epoch=TRAIN_STEPS,
                epochs=EPOCHS,
                validation_data=test_data_gen,
                validation_steps=VAL_STEPS,
                # callbacks = [cp_callback]
                )

### Ploting
loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

print(history.history)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title('Accuray')
plt.plot(epochs, history.history['row_out_accuracy'], 'g', label='row_out_accuracy')
plt.plot(epochs, history.history['col_out_accuracy'], 'g', label='col_out_accuracy')
plt.plot(epochs, history.history['val_row_out_accuracy'], 'g', label='val_row_out_accuracy')
plt.plot(epochs, history.history['val_col_out_accuracy'], 'g', label='val_col_out_accuracy')
plt.grid(True)
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.title('Loss')
plt.plot(epochs, history.history['loss'], 'r', label='loss')
plt.plot(epochs, history.history['row_out_loss'], 'r', label='row_out_loss')
plt.plot(epochs, history.history['col_out_loss'], 'r', label='col_out_loss')
plt.plot(epochs, history.history['val_loss'], 'g', label='val_loss')
plt.plot(epochs, history.history['val_row_out_loss'], 'g', label='val_row_out_loss')
plt.plot(epochs, history.history['val_col_out_loss'], 'g', label='val_col_out_loss')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()


# ### 결과 출력을 위한 함수
def Make_Result_Plot(suptitle, data, label, y_max):
    fig_result, ax_result = plt.subplots(2, 5, figsize=(18, 7))
    fig_result.suptitle(suptitle)
    for idx in range(10):
        ax_result[idx//5][idx%5].imshow(data[idx],cmap="binary")
        ax_result[idx//5][idx%5].set_title("test_data[{}] (label : {} / y : {})".format(idx, label[idx], y_max[idx]))

### 학습 후 상황
y_out = model_out.predict(test_data)
y_max = np.argmax(y_out, axis=1).reshape((-1, 1))
Make_Result_Plot("After Training", test_data, test_labels, y_max)