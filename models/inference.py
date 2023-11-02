import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from glob import glob

from params import Params
from create_model import Create_model


### Parameters
params = Params()
PRE_MODEL_PATH = ""  # saved model path or params.PRE_MODEL_PATH
TEST_IMG_PATH = ""  # test image path or params.TEST_IMG_PATH
RESIZED_WIDTH = params.RESIZED_WIDTH
RESIZED_HEIGHT = params.RESIZED_HEIGHT


### Load test data
test_images = glob(os.path.join(TEST_IMG_PATH, "*.jpg"))
test_images = np.array([(cv2.resize(cv2.cvtColor(cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (RESIZED_WIDTH,RESIZED_HEIGHT))) for x in test_images])
test_images = test_images / 255.0


### Load model
model_creator = Create_model()
model_out = model_creator.create_model()
if PRE_MODEL_PATH:
    model_out.load_weights(PRE_MODEL_PATH)


### output 생성하여 화면 컨트롤
for test_image in test_images:
    y_out = model_out.predict(test_image)
    y_max = np.argmax(y_out, axis=1)
    print('predict: ', y_max)