import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import tflite_runtime.interpreter as tflite
import numpy as np


def load_data(file):
    npzfile = np.load(file)
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]
    val_eye_left2 = npzfile["val_eye_left2"]
    val_eye_right2 = npzfile["val_eye_right2"]
    val_face2 = npzfile["val_face2"]
    val_face_mask2 = npzfile["val_face_mask2"]
    return [val_eye_left, val_eye_right, val_face, val_face_mask, val_y],[val_eye_left2, val_eye_right2, val_face2, val_face_mask2]


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
    
def prepare_data2(data):
    # 전처리
    eye_left, eye_right, face, face_mask = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    return [eye_left, eye_right, face, face_mask]
    
DATA_PATH = './data/eye_tracker_train_and_val_lite.npz'
test_data, test_data2 = load_data(DATA_PATH)
test_data = prepare_data(test_data)
test_label = test_data[-1]
test_data = test_data[:-1]

test_data2 = prepare_data2(test_data2)

model_path = "./converted_final.tflite"
interpreter=tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details() 
floating_model = input_details[0]['dtype'] == np.float32

output_list = []
output_list2 = []

for j in range(len(test_data[0])):
    for i in range(4):
        interpreter.set_tensor(input_details[i]['index'], np.expand_dims(test_data[3-i][j], axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output_list.append(output[0])
    
for j in range(len(test_data2[0])):
    for i in range(4):
        interpreter.set_tensor(input_details[i]['index'], np.expand_dims(test_data2[3-i][j], axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output_list2.append(output[0])

#============= Plot ==============
def Make_Result_Plot(suptitle, data, label, y_max, pretrained=True):
    i = []
    if pretrained:
        i = [4,5,14,15,16,23,24,38,42,45,47,54,55]
    else:
        i = [i for i in range(0,15)]
    for idx in i:
        print(idx)
        fig_result, ax_result = plt.subplots(1,2,figsize=(10,6), gridspec_kw={'wspace': 0.5, 'hspace': 0})
        fig_result.suptitle(suptitle)
        
        ax_result[0].set_xticks([])
        ax_result[0].set_yticks([])
        
        ax_result[1].set_xlim(20,-20)
        ax_result[1].set_ylim(-20,20)
        
        ax_result[1].set_aspect('equal')
        
        ax_result[0].imshow(data[2][idx] * 3,cmap="binary")
        
        if pretrained:
          ax_result[1].scatter(label[idx,0], label[idx,1], color="green", label="label")
        ax_result[1].scatter(y_max[idx][0], y_max[idx][1], color="blue", label="predict")
        
        ax_result[1].set_xlabel('x')
        ax_result[1].set_ylabel('y')
        
        plt.show()
Make_Result_Plot("Predict", test_data, test_label, output_list, pretrained=True)
Make_Result_Plot("Predict", test_data2, None, output_list2, pretrained=False)
