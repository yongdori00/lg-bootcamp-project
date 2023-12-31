### Parameters

# Data Params
class Params:
    def __init__(self):
        self.IMG_WIDTH = 160  # 한 쪽 눈 이미지의 너비
        self.RESIZED_WIDTH = self.IMG_WIDTH
        self.IMG_HEIGHT = 60  # 한 쪽 눈 이미지의 높이
        self.RESIZED_HEIGHT = self.IMG_HEIGHT
        self.IMG_CHANNELS = 3  # gray scale
        self.IMG_PATH = "./new_cropped"
        self.CSV_PATH = "./output.csv"
        self.TEST_IMG_PATH = ""
        self.TEST_CSV_PATH = ""
        self.DF_XCOL = "name"
        self.DF_YCOL = ["row", "column"]

        # Grid Params
        self.GRID_ROWS = 2
        self.GRID_COLS = 3
        self.LABELS = self.GRID_ROWS * self.GRID_COLS

        # Model Params
        self.BATCH_N = 16
        self.EPOCHS = 100
        # self.TRAIN_STEPS = int(len(train_data)/self.BATCH_N)
        self.VAL_STEPS = 30
        self.DENSE_UNITS = 512
        self.PRE_MODEL_PATH = None  # pretrained model path(.ckpt) or None
        self.SAVE_MODEL_PATH = "./checkpoints"