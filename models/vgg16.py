import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from create_model import Create_model
from params import Params


### Parameters
params = Params()
IMG_PATH = params.IMG_PATH
CSV_PATH = params.CSV_PATH
DF_XCOL = params.DF_XCOL
DF_YCOL = params.DF_YCOL
RESIZED_WIDTH = params.RESIZED_WIDTH
RESIZED_HEIGHT = params.RESIZED_HEIGHT
BATCH_N = params.BATCH_N
EPOCHS = params.EPOCHS
VAL_STEPS = params.VAL_STEPS
PRE_MODEL_PATH = params.PRE_MODEL_PATH
SAVE_MODEL_PATH = params.SAVE_MODEL_PATH


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
                      target_size=(RESIZED_HEIGHT, RESIZED_WIDTH),                     # 변경될 이미지 데이터의 크기
                      class_mode = 'multi_output', # Label을 숫자로 표기
                      subset='training'           # Training 용 데이터: 전체의 80%
                      )

test_data_gen = image_gen.flow_from_dataframe(
                      dataframe=df,         # 디렉터리 지정
                      directory=IMG_PATH,
                      x_col=DF_XCOL,
                      y_col=DF_YCOL,
                      batch_size=BATCH_N,         # 한번에 생성할 데이터의 크기 설정
                      target_size=(RESIZED_HEIGHT, RESIZED_WIDTH),                     # 변경될 이미지 데이터의 크기
                      class_mode = 'multi_output', # Label을 숫자로 표기
                      subset='validation'         # Testing 용 데이터: 전체의 20%
                      )

print("******************************")
train_data_t, train_labels_t = next(train_data_gen)
print(np.array(train_data_t).shape)
print(np.array(train_labels_t).shape)
print(train_labels_t)


### Load model
model_creator = Create_model()
model_out = model_creator.create_model()
if PRE_MODEL_PATH:
    model_out.load_weights(PRE_MODEL_PATH)


### checkpoint 지정
checkpoint_path = os.path.join(SAVE_MODEL_PATH, "checkpoint-{epoch:04d}.ckpt")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)


### Training
# %%time
history = model_out.fit(
                train_data_gen,
                # steps_per_epoch=TRAIN_STEPS,
                epochs=EPOCHS,
                validation_data=test_data_gen,
                validation_steps=VAL_STEPS,
                callbacks = [checkpoint_callback]
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


### 결과 출력을 위한 함수
# def Make_Result_Plot(suptitle, data, label, y_max):
#     fig_result, ax_result = plt.subplots(2, 5, figsize=(18, 7))
#     fig_result.suptitle(suptitle)
#     for idx in range(10):
#         ax_result[idx//5][idx%5].imshow(data[idx],cmap="binary")
#         ax_result[idx//5][idx%5].set_title("test_data[{}] (label : {} / y : {})".format(idx, label[idx], y_max[idx]))

# ### 학습 후 상황
# y_out = model_out.predict(test_data)
# y_max = np.argmax(y_out, axis=1).reshape((-1, 1))
# Make_Result_Plot("After Training", test_data, test_labels, y_max)