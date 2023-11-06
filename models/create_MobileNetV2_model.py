import tensorflow as tf
from models.params import Params

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2



### Create new model
class Create_model:
    def __init__(self):
        self.params = Params()
    
    def create_model(self):
        ### Load MobileNetV2 Model
        model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(self.params.RESIZED_HEIGHT, self.params.RESIZED_WIDTH, self.params.IMG_CHANNELS))
            )
        # print('original MobileNetV2')
        # print(model.summary())


        ### Set trainable option
        # for layer in model.layers[:-4]:
            # layer.trainable = False
        # for layer in model.layers:
        #     print(layer, layer.trainable)


        ### Layer 추가
        # x = Flatten()(model.layers[-1].output)
        x = GlobalAveragePooling2D()(model.layers[-1].output)
        x = Dropout(rate=0.2)(x)
        # x = Dense(self.params.DENSE_UNITS, activation='relu')(x)
        # x = BatchNormalization()(x)

        # 첫번째 출력: ROW
        # row_out = Dense(self.params.GRID_ROWS, activation='softmax', name='row_out')(x)

        # 두번째 출력: COL
        col_out = Dense(self.params.GRID_COLS, activation='softmax', name='col_out')(x)

        # 최종 모델
        # model_out = tf.keras.models.Model(inputs=model.input, outputs=[row_out, col_out])
        model_out = tf.keras.models.Model(inputs=model.input, outputs=col_out)
        # print('fine tuned model')
        # print(model_out.summary())


        ### Loss function
        # loss_weights = {'row_out': 1.0, 'col_out': 1.0}  # 각 출력에 대한 가중치
        # loss = {'row_out': 'categorical_crossentropy', 'col_out': 'categorical_crossentropy'}  # 각 출력에 대한 손실 함수


        ### Model compile
        model_out.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                # loss_weights=1.0,
                metrics=['accuracy'])
        
        return model_out

if __name__ == "__main__":
    model_creator = Create_model()
    model_out = model_creator.create_model()
