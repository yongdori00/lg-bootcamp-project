### Parameters

# Data Params
class Params:
    def __init__(self):
        self.IMG_WIDTH = 212
        self.RESIZED_WIDTH = self.IMG_WIDTH
        self.IMG_HEIGHT = 320
        self.RESIZED_HEIGHT = self.IMG_HEIGHT
        self.IMG_CHANNELS = 3
        self.IMG_PATH = "./cropped"
        self.CSV_PATH = "./output.csv"
        self.TEST_IMG_PATH = ""
        self.TEST_CSV_PATH = ""
        self.DF_XCOL = "name"
        self.DF_YCOL = ["row", "column"]

        # Grid Params
        self.GRID_ROWS = 6
        self.GRID_COLS = 8
        self.LABELS = self.GRID_ROWS * self.GRID_COLS

        # Model Params
        self.BATCH_N = 32
        self.EPOCHS = 3
        # self.TRAIN_STEPS = int(len(train_data)/self.BATCH_N)
        self.VAL_STEPS = 6
        self.DENSE_UNITS = 1024
        self.PRE_MODEL_PATH = None  # pretrained model path or None
        self.SAVE_MODEL_PATH = "./checkpoints"