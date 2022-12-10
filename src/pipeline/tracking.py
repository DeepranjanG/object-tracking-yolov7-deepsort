import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.model_ingestion import ModelIngestion
from src.components.model_loading import ModelLoading
from src.components.data_transformation import DataTransformation
from src.components.object_tracking import ObjectTracking
from src.components.pusher import Pusher
from src.entity.config_entity import ModelIngestionConfig, ModelLoadingConfig, DataTransformationConfig, ObjectTrackingConfig, PusherConfig
from src.entity.artifact_entity import ModelIngestionArtifacts, ModelLoadingArtifacts, DataTransformationArtifacts, ObjectTrackingArtifacts, PusherArtifacts


class TrackingPipeline:
    def __init__(self):
        self.model_ingestion_config = ModelIngestionConfig()
        self.model_loading_config = ModelLoadingConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.object_tracking_config = ObjectTrackingConfig()
        self.pusher_config = PusherConfig()

    def start_model_ingestion(self) -> ModelIngestionArtifacts:
        logging.info("Entered the start_model_ingestion method of TrackingPipeline class")
        try:
            logging.info("Getting the weights from GCLoud Storage bucket")
            model_ingestion = ModelIngestion(
                model_ingestion_config=self.model_ingestion_config)

            model_ingestion_artifacts = model_ingestion.initiate_model_ingestion()
            logging.info("Got the weights from GCLoud Storage")
            logging.info("Exited the start_model_ingestion method of TrackingPipeline class")
            return model_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_loading(self, model_ingestion_artifacts: ModelIngestionArtifacts) -> ModelLoadingArtifacts:
        logging.info("Entered the start_model_loading method of TrackingPipeline class")
        try:
            logging.info("load weights and save into models")
            model_loading = ModelLoading(
                model_loading_config=self.model_loading_config,
                model_ingestion_artifacts=model_ingestion_artifacts
            )
            model_loading_artifacts = model_loading.initiate_model_loading()
            logging.info("Got the weights and save into models")
            logging.info("Exited the start_model_loading method of TrackingPipeline class")
            return model_loading_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self, model_loading_artifacts: ModelLoadingArtifacts) -> DataTransformationArtifacts:
        logging.info("Entered the start_model_loading method of TrackingPipeline class")
        try:
            logging.info("load weights and save into models")
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                model_loading_artifacts=model_loading_artifacts
            )
            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info("Got the weights and save into models")
            logging.info("Exited the start_model_loading method of TrackingPipeline class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_object_tracking(self, data_transformation_artifacts: DataTransformationArtifacts,
                              model_loading_artifacts: ModelLoadingArtifacts) -> ObjectTrackingArtifacts:
        logging.info("Entered the start_object_tracking method of TrackingPipeline class")
        try:
            logging.info("load model, dataset and save tracked video")
            object_tracking = ObjectTracking(
                object_tracking_config=self.object_tracking_config,
                data_transformation_artifacts=data_transformation_artifacts,
                model_loading_artifacts=model_loading_artifacts
            )
            object_tracking_artifacts = object_tracking.initiate_object_tracking()
            logging.info("Got the weights and save into models")
            logging.info("Exited the start_model_loading method of TrackingPipeline class")
            return object_tracking_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_pusher(self, object_tracking_artifacts: ObjectTrackingArtifacts) -> PusherArtifacts:
        logging.info("Entered the start_pusher method of TrackingPipeline class")
        try:
            logging.info("Loaded tracked video")
            pusher = Pusher(
                pusher_config=self.pusher_config,
                object_tracking_artifacts=object_tracking_artifacts
            )
            pusher_artifacts = pusher.initiate_pusher()
            logging.info("Pushed tracked video on Gcloud Storage")
            logging.info("Exited the start_pusher method of TrackingPipeline class")
            return pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrackingPipeline class")
        try:
            model_ingestion_artifacts = self.start_model_ingestion()

            model_loading_artifacts = self.start_model_loading(
                model_ingestion_artifacts=model_ingestion_artifacts
            )

            data_transformation_artifacts = self.start_data_transformation(
                model_loading_artifacts=model_loading_artifacts
            )

            object_tracking_artifacts = self.start_object_tracking(
                data_transformation_artifacts=data_transformation_artifacts,
                model_loading_artifacts=model_loading_artifacts
            )

            pusher_artifacts = self.start_pusher(
                object_tracking_artifacts=object_tracking_artifacts
            )

            logging.info("Exited the run_pipeline method of TrackingPipeline class")
        except Exception as e:
            raise CustomException(e, sys) from e
