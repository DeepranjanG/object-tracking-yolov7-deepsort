import sys
from numpy import random
from src.logger import logging
from src.exception import CustomException
from src.constants import IMG_SIZE, SOURCE
from src.ml.load_images import LoadImages
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import ModelLoadingArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                 model_loading_artifacts:ModelLoadingArtifacts):
        """
        :param data_transformation_config: Configuration for data transformation
        :param model_loading_artifacts: Artifacts for model loading
        """
        self.data_transformation_config = data_transformation_config
        self.model_loading_artifacts = model_loading_artifacts

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Method Name :   initiate_data_transformation
        Description :   This function initiates a data transformation steps

        Output      :   Returns data transformation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the initiate_data_transformation method of Data transformation class")
        try:
            model = self.model_loading_artifacts.model_object
            logging.info("Loaded torch script model")

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            logging.info("Get names and colors")

            dataset = LoadImages(SOURCE, img_size=IMG_SIZE, stride=self.model_loading_artifacts.stride)

            logging.info("Load and conversion of dataset done")

            data_transformation_artifacts = DataTransformationArtifacts(dataset_obj=dataset,
                                                                        class_name=names,
                                                                        class_colors=colors)
            logging.info(f"Model ingestion artifact: {data_transformation_artifacts}")
            logging.info("Exited the initiate_data_transformation method of Data transformation class")
            return data_transformation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
