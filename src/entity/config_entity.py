import os
from src.constants import *
from dataclasses import dataclass


@dataclass
class ModelIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.WEIGHTS_DIR: str = WEIGHTS_DIR
        self.MODEL_NAME: str = MODEL_NAME
        self.MODEL_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_INGESTION_ARTIFACTS_DIR)
        self.WEIGHTS_DIR_PATH: str = os.path.join(self.MODEL_INGESTION_ARTIFACTS_DIR, self.WEIGHTS_DIR)
        self.MODEL_PATH: str = os.path.join(self.WEIGHTS_DIR_PATH, self.MODEL_NAME)


@dataclass
class ModelLoadingConfig:
    def __init__(self):
        self.MODEL_NAME = MODEL_NAME
        self.TRACED_MODEL = TRACED_MODEL
        self.MODEL_LOADING_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_LOADING_ARTIFACTS_DIR)
        self.MODEL_PATH: str = os.path.join(self.MODEL_LOADING_ARTIFACTS_DIR, self.MODEL_NAME)
        self.TRACED_MODEL_PATH: str = os.path.join(self.MODEL_LOADING_ARTIFACTS_DIR, self.TRACED_MODEL)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)


@dataclass
class ObjectTrackingConfig:
    def __init__(self):
        self.OBJECT_TRACKING_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, OBJECT_TRACKING_ARTIFACTS_DIR)
        self.DETECT_DIR: str = os.path.join(self.OBJECT_TRACKING_ARTIFACTS_DIR, DETECT_DIR)


@dataclass
class PusherConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.DIR_NAME: str = TRACKED_DIR


