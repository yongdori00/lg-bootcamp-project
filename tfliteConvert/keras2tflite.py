import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import tensorflow as tf
from models.create_model import Create_model


### 변환할 모델
org_model = "./data/checkpoints/checkpoint-0030.ckpt"  # checkpoint model (.ckpt)
### 변환된 후 모델 저장 위치
save_dir = "./tfliteConvert"
### 저장할 모델 이름
new_model_name = "test_tfliteModel"


### Load model
model_creator = Create_model()
new_model = model_creator.create_model()
# new_model.load_weights(org_model)

### Convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = converter.convert()
open(os.path.join(save_dir, new_model_name + '.tflite'), 'wb').write(tflite_model)