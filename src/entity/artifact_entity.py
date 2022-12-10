from dataclasses import dataclass


# Model ingestion artifacts
@dataclass
class ModelIngestionArtifacts:
    weights_path: str

    def to_dict(self):
        return self.__dict__


# Model loading artifacts
@dataclass
class ModelLoadingArtifacts:
    model_object: object
    stride: int
    imgsz: int

    def to_dict(self):
        return self.__dict__


# Data transformation artifacts
@dataclass
class DataTransformationArtifacts:
    dataset_obj: object
    class_name: list
    class_colors: list

    def to_dict(self):
        return self.__dict__


# Object tracking artifacts
@dataclass
class ObjectTrackingArtifacts:
    artifacts_path: str
    output_path: str

    def to_dict(self):
        return self.__dict__


# Artifacts pusher
@dataclass
class PusherArtifacts:
    bucket_name: str
    dir_name: str

    def to_dict(self):
        return self.__dict__
