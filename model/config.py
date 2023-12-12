# Configure the parameters of our project

import os
import torch
import platform

# The device to be used for training and evaluation (Windows)
if platform.system() == "Darwin":
    if torch.backends.mps.is_available():
        DEVICE = torch.device("cpu")
    else:
        print ("MPS device for MacOS not found.")
    WORKING_DIRECTORY_PATH = "./SICOM_DeepLearning/Semantic_Segmentation_U-NET/"

elif platform.system() == 'Linux':
    WORKING_DIRECTORY_PATH = "/home/conversb/Semantic_Segmentation_U-NET/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKING_DIRECTORY_PATH = ""

# Define the test split which will separate our dataset into train and test
TEST_SPLIT = 0.15

# List of gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

# Vizualization parameters
# True if you want print graphs during the training
MODE_VISUALIZATION = True
VISUALIZATION_DIM = 16  # TODO: There are some buts, it needed to avoid multiples of 4

# Test or Train the model
TYPE_PROCESS = "test"  #value: {"train", "test"}
ID_SESSION = "train_12_12_23_part-4"

# Activate Parallelism
ACTIVATE_PARALLELISM = False
NBR_WORKERS = 24
NBR_GPU = 3

# Early Stopping
EARLY_STOPPING_ACTIVATE = False
PATIENCE = 5

# Define some model parameters
ENC_CHANNELS= (3, 16, 32, 64, 128)
DEC_CHANNELS = (128, 64, 32, 16)
NBR_CLASSES = 24
INPUT_IMAGE_HEIGHT = 64
INPUT_IMAGE_WIDTH = 64
BATCH_SIZE = 64
NUM_EPOCHS = 1
INIT_LR = 0.001 # Initial learning rate
THRESHOLD_TYPE = "mean"
SELECTED_IMAGE_TEST = 10

# Define different paths
# The images dataset
IMAGE_DATASET_PATH = os.path.join(WORKING_DIRECTORY_PATH, "dataset", "semantic_drone_dataset", "original_images")
# The RGB color masks of the dataset
MASK_DATASET_PATH = os.path.join(WORKING_DIRECTORY_PATH, "dataset", "semantic_drone_dataset", "RGB_color_image_masks")
# Labels
LABEL_PATH = os.path.join(WORKING_DIRECTORY_PATH, "dataset", "semantic_drone_dataset", "class_dict_seg.csv")
# The output directory
BASE_OUTPUT = os.path.join(WORKING_DIRECTORY_PATH, "output")
# The testing image path
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, ID_SESSION, "test_paths.txt"])
# The model training plot path
PLOT_TRAIN_PATH = os.path.sep.join([BASE_OUTPUT, ID_SESSION, "train_plots"])
# The model test plot path
PLOT_TEST_PATH = os.path.sep.join([BASE_OUTPUT, ID_SESSION, "test_plots"])
# The output serialized model path to save it
MODEL_PATH = os.path.join(BASE_OUTPUT, ID_SESSION, "unet_tgs_salt.pth")

# List of image types
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# If we will be pinning memory during data loading
PIN_MEMORY = True if( DEVICE == "cuda" or DEVICE == "mps") else False
