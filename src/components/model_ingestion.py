import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import ModelIngestionConfig
from src.entity.artifact_entity import ModelIngestionArtifacts


class ModelIngestion:

    def __init__(self, model_ingestion_config: ModelIngestionConfig):
        """
        :param model_ingestion_config: Configuration for model ingestion
        """
        self.model_ingestion_config = model_ingestion_config
        self.gcloud = GCloudSync()

    def get_weights_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_weights_from_gcloud method of Model ingestion class")

            os.makedirs(self.model_ingestion_config.WEIGHTS_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.model_ingestion_config.BUCKET_NAME,
                                                self.model_ingestion_config.WEIGHTS_DIR,
                                                self.model_ingestion_config.MODEL_INGESTION_ARTIFACTS_DIR,
                                                )

            logging.info("Exited the get_weights_from_gcloud method of Model ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_models_folder_from_gcloud(self) -> None:
        try:
            logging.info("Entered the get_models_folder_from_gcloud method of Model ingestion class")

            os.makedirs(self.model_ingestion_config.WEIGHTS_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(self.model_ingestion_config.BUCKET_NAME, "models", os.getcwd())

            self.gcloud.sync_folder_from_gcloud(self.model_ingestion_config.BUCKET_NAME, "utils", os.getcwd())

            logging.info("Exited the get_models_folder_from_gcloud method of Model ingestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_ingestion(self) -> ModelIngestionArtifacts:
        """
        Method Name :   initiate_model_ingestion
        Description :   This function initiates a model ingestion steps

        Output      :   Returns model ingestion artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the initiate_model_ingestion method of Model ingestion class")
        try:
            self.get_weights_from_gcloud()
            logging.info("Fetched the weights from Gcloud Storage bucket")

            self.get_models_folder_from_gcloud()
            logging.info("Fetched the models & utils folder from Gcloud Storage bucket")

            model_ingestion_artifacts = ModelIngestionArtifacts(weights_path=self.model_ingestion_config.MODEL_PATH)
            logging.info(f"Model ingestion artifact: {model_ingestion_artifacts}")

            logging.info("Exited the initiate_model_ingestion method of Model ingestion class")
            return model_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
