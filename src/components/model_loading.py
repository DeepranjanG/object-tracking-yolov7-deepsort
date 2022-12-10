import os
import sys
import torch
import torch.nn as nn
from src.ml.conv import Conv
from src.logger import logging
from src.constants import DEVICE, IMG_SIZE, HALF
from src.exception import CustomException
from src.utils.general import check_img_size
from src.utils.torch_utils import TracedModel
from src.entity.config_entity import ModelLoadingConfig
from src.entity.artifact_entity import ModelIngestionArtifacts, ModelLoadingArtifacts


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class ModelLoading:
    def __init__(self, model_loading_config: ModelLoadingConfig, model_ingestion_artifacts: ModelIngestionArtifacts):
        """
        :param model_loading_config: Configuration for model loading
        :param model_ingestion_artifacts: Artifacts for model ingestion
        """
        self.model_loading_config = model_loading_config
        self.model_ingestion_artifacts = model_ingestion_artifacts

    def compatibility_updates(self, model):
        try:
            for m in model.modules():
                if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True  # pytorch 1.7.0 compatibility
                elif type(m) is nn.Upsample:
                    m.recompute_scale_factor = None  # torch 1.11.0 compatibility
                elif type(m) is Conv:
                    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        except Exception as e:
            raise CustomException(e, sys) from e

    def ensemble_model_check(self, model, weights):
        try:
            if len(model) == 1:
                logging.info("Return single model")
                return model[-1]  # return model
            else:
                print('Ensemble created with %s\n' % weights)
                for k in ['names', 'stride']:
                    setattr(model, k, getattr(model[-1], k))
                    logging.info("Return ensemble model")
                return model  # return ensemble
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_loading(self) -> ModelLoadingArtifacts:
        """
        Method Name :   initiate_model_loading
        Description :   This function initiates a model loading steps

        Output      :   Returns model loading artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the initiate_model_loading method of model loading class")
        try:
            with torch.no_grad():
                model = Ensemble()
                logging.info("Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a")
                weights = self.model_ingestion_artifacts.weights_path
                ckpt = torch.load(weights, map_location=DEVICE)
                logging.info("loaded model")

                model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
                self.compatibility_updates(model)
                logging.info("Compatibility updated done")
                model = self.ensemble_model_check(model, weights)

                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size

                model = TracedModel(model=model, device=DEVICE, img_size=IMG_SIZE)
                logging.info("Converted PyTorch model to a Torch Script")

                if HALF:
                    model.half()  # to FP16
                logging.info("Converting model weights to half precision. Training in fp16 will be faster")

            model_loading_artifacts = ModelLoadingArtifacts(model_object=model, stride=stride, imgsz=imgsz)
            logging.info(f"Model ingestion artifact: {model_loading_artifacts}")

            logging.info("Exited the initiate_model_loading method of Model loading class")
            return model_loading_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
