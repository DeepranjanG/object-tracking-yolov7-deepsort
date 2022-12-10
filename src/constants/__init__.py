import os
import torch
from datetime import datetime

# Common constants
IMG_SIZE = 512
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if use_cuda else "cpu")
# DEVICE = torch.device("cpu")
HALF = DEVICE.type != "cpu"

APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Model ingestion constants
MODEL_INGESTION_ARTIFACTS_DIR = 'ModelIngestionArtifacts'
BUCKET_NAME = 'object-tracking'
WEIGHTS_DIR = 'Yolov7'

# Model Loading constants
MODEL_LOADING_ARTIFACTS_DIR = "ModelLoadingArtifacts"
MODEL_NAME = "yolov7.pt"
TRACED_MODEL = "traced_model.pt"

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
SOURCE = "soccer.mp4"
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv', 'webm']  # acceptable video suffixes

# Object tracking constants
OBJECT_TRACKING_ARTIFACTS_DIR = "ObjectTrackingArtifacts"
DETECT_DIR = "detect"

# Pusher constants
TRACKED_DIR = os.path.join("detect", TIMESTAMP)





