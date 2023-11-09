import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from params import Params
from itracker_keras import ITrackerModel

def Make_Result_Plot(suptitle, data, label):
    fig_result, ax_result = plt.subplots(2,5,figsize=(24, 10))
    fig_result.suptitle(suptitle)
    for idx in range(10):
        ax_result[idx//5][idx%5].imshow(data[idx], cmap="binary")
        ax_result[idx//5][idx%5].set_title("test_data[{}] (label : {} / y :)".format(idx, label[idx]))
    plt.show()


def load_data(file):
    npzfile = np.load(file)
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]
    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]

def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255. # scaling
    data = data - np.mean(data, axis=0) # normalizing
    return np.reshape(data, shape)


def prepare_data(data):
    # 전처리
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]

model_path = "./checkpoints/checkpoint-0093.ckpt"
# model = ITrackerModel()
# model.load_weights(filepath=model_path)
model = tf.keras.models.load_model("./my_model")

DATA_PATH = './data/eye_tracker_train_and_val.npz'
train_data, test_data = load_data(DATA_PATH)
test_data = prepare_data(test_data)
test_label = test_data[-1]
test_data = test_data[:-1]
print("Test Data loaded")

eye_left = test_data[0][30:40]
eye_right = test_data[1][30:40]
face = test_data[2][30:40]
face_mask = test_data[3][30:40]

# print(eye_left, eye_left.shape)
# print(eye_right, eye_right.shape)
# print(face, face.shape)
# print(face_mask, face_mask.shape)

print(np.array(eye_left[0]).shape)
print(np.array(eye_right[0]).shape)
print(np.array(face[0]).shape)
print(np.array(face_mask[0]).shape)


# numpy 값. 줄임없이 모두 표시하는 옵션
# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=200)
print(face_mask)
for f in face_mask:
    f = np.array(f, dtype=np.int64)
    f = np.reshape(f, (25, 25))
    for i in range(25):
        print(f[i])
    print()
# Make_Result_Plot("Result", face_mask, test_label)
Make_Result_Plot("Result", face, test_label[30:40])
Make_Result_Plot("Result", eye_left, test_label[30:40])
Make_Result_Plot("Result", eye_right, test_label[30:40])
print("plot")

# eye_left = test_data[0][:10]
# eye_right = test_data[1][:10]
# face = test_data[2][:10]
# face_mask = test_data[3][:10]
predictions = model.predict(test_data)
labels = test_label[30:40]
for i in range(10):
    print(predictions[i])
    print(labels[i])
    print()
# print(predictions)
# y_out = model.predict([eye_left, eye_right, face, face_mask])

eye_left = test_data[0][31:32]
eye_right = test_data[1][31:32]
face = test_data[2][31:32]
face_mask = test_data[3][31:32]
test_label = test_label[31:32]

test_data = [eye_left, eye_right, face, face_mask]
loss, accuracy = model.evaluate(test_data, test_label)
print()
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')



# plt.imshow(img)
# plt.show() # 이미지 그리기
# img = image.img_to_array(img)                          # 이미지를 ndarray로 변경: (224,224,3)
# img2 = img.copy()
# img = np.expand_dims(img, axis=0)                      # 차원 확장: (1,224,224,3)
# img = preprocess_input(img)                            # VGG-16 모델 입력에 맞도록 전처리
# print(img.shape)
#
#
#
#
# results = model_out.predict(img)
# print("result:", results, np.argmax(results, axis=1))
#
# # Grad cam
# model_grad = tf.keras.models.Model([model_out.inputs],
#                                    [model_out.get_layer('Conv_1').output, model_out.output])
#
# with tf.GradientTape() as tape:
#     conv_outputs, predict = model_grad(img) # 모델을 돌려서 output 2개 출력
#     class_out = predict[:, np.argmax(predict[0])] # 타겟 Class 번호
#
# output = conv_outputs[0] # 타겟 Feature map
# grads = tape.gradient(class_out, conv_outputs)[0] # 편미분 결과
# print(grads.shape, output.shape)
#
# weights = tf.reduce_mean(grads, axis=(0, 1)) # GAP 연산
# cam = np.zeros(output.shape[0:2], dtype=np.float32) # CAM 초기화
#
# for index, w in enumerate(weights):
#     cam += w * output[:, :, index] # Grad-CAM 연산
# print(cam.shape, weights.shape)
#
# import cv2
# # img = cv2.imread(img_path)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # plt.imshow(img)
# # plt.show()
# # print(img.shape, type(img))
#
# cam = cv2.resize(cam.numpy(), (img2.shape[1], img2.shape[0]))
# cam = np.maximum(cam, 0) # ReLU 연산: 0 이하는 0, 0 이상은 그대로
# heatmap = (cam - cam.min()) / (cam.max() - cam.min()) # 0 ~ 1로 변환
# heatmap = np.uint8(255 * heatmap) # 0 ~ 255로 변환
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Color map 적용
# heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # BGR을 RGB로
#
# plt.imshow(heatmap)
# plt.show()
# print(heatmap.shape)
#
# output_image = cv2.addWeighted(img2.astype('uint8'), 1, # 이미지 100%
#                                heatmap, 0.7, 0)
# plt.imshow(output_image)
# plt.show()
# plt.axis('off')