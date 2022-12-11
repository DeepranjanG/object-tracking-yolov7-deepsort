import os
import sys
import cv2
import torch
import time
import numpy as np
from pathlib import Path
from src.ml.sort import Sort
from src.logger import logging
from src.utils import draw_boxes
from src.constants import DEVICE, HALF
from src.exception import CustomException
from src.utils.torch_utils import time_synchronized
from src.entity.config_entity import ObjectTrackingConfig
from src.utils.general import non_max_suppression, scale_coords
from src.entity.artifact_entity import ObjectTrackingArtifacts, DataTransformationArtifacts, ModelLoadingArtifacts
torch.cuda.empty_cache()


class ObjectTracking:
    def __init__(self, object_tracking_config: ObjectTrackingConfig,
                 data_transformation_artifacts: DataTransformationArtifacts,
                 model_loading_artifacts: ModelLoadingArtifacts):
        """
        :param object_tracking_config: Configuration for object tracking
        :param data_transformation_artifacts: Artifacts for data transformation
        """
        self.object_tracking_config = object_tracking_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_loading_artifacts = model_loading_artifacts
        self.sort_tracker = Sort()

    def initiate_object_tracking(self) -> ObjectTrackingArtifacts:
        """
        Method Name :   initiate_object_tracking
        Description :   This function initiates a object tracking steps

        Output      :   Returns object tracking artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the initiate_object_tracking method of Object tracking class")
        try:
            os.makedirs(self.object_tracking_config.DETECT_DIR, exist_ok=True)

            model = self.model_loading_artifacts.model_object
            model.to(DEVICE)

            logging.info("Loaded torch script model")

            dataset = self.data_transformation_artifacts.dataset_obj
            logging.info("Loading dataset from data transformation artifacts")

            names = self.data_transformation_artifacts.class_name
            logging.info("Loading class name from data transformation artifacts")

            vid_path, vid_writer = None, None
            t0 = time.time()

            for path, img, im0s, vid_cap in dataset:
                try:
                    img = torch.from_numpy(img).to(DEVICE)

                    img = img.half() if HALF else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    logging.info("Iterating through complete data and perform scaling")

                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]

                    logging.info(f"Get the image height: {old_img_h}, image weight: {old_img_w}, image bias: {old_img_b}")
                    logging.info("Begin inference")
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]
                    t2 = time_synchronized()

                    logging.info("Applying non-maximum separation to prediction")
                    pred = non_max_suppression(pred)
                    t3 = time_synchronized()

                    logging.info("Process detections, detections per image")
                    for i, det in enumerate(pred):
                        try:
                            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                            p = Path(p)  # to Path
                            save_path = os.path.join(self.object_tracking_config.DETECT_DIR, p.name)
                            txt_path = str(os.path.join(self.object_tracking_config.DETECT_DIR, "labels", p.stem)) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            if len(det):
                                logging.info("Rescale boxes from img_size to im0 size")
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                                logging.info("Print results")
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                logging.info("USING TRACK FUNCTION")
                                logging.info("pass an empty array to sort")
                                dets_to_sort = np.empty((0, 6))

                                logging.info("NOTE: We send in detected object class too")
                                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                    dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
                                logging.info("Running SORT")
                                tracked_dets = self.sort_tracker.update(dets_to_sort)
                                tracks = self.sort_tracker.getTrackers()
                                txt_str = ""

                                logging.info("loop over tracks")
                                for track in tracks:
                                    try:
                                        [cv2.line(im0, (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                                                  (int(track.centroidarr[i + 1][0]), int(track.centroidarr[i + 1][1])),
                                                  (255, 0, 0), thickness=2)
                                         for i, _ in enumerate(track.centroidarr)
                                         if i < len(track.centroidarr) - 1]
                                    except Exception as e:
                                        raise CustomException(e, sys) from e

                                logging.info("draw boxes for visualization")
                                if len(tracked_dets) > 0:
                                    bbox_xyxy = tracked_dets[:, :4]
                                    identities = tracked_dets[:, 8]
                                    categories = tracked_dets[:, 4]
                                    draw_boxes(im0, bbox_xyxy, identities, categories, names)

                            logging.info("Print time (inference + NMS)")
                            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                            logging.info("Save video results")
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)
                        except Exception as e:
                            raise CustomException(e, sys) from e
                except Exception as e:
                    raise CustomException(e, sys) from e
            print(f'Done. ({time.time() - t0:.3f}s)')

            object_tracking_artifacts = ObjectTrackingArtifacts(
                artifacts_path=self.object_tracking_config.OBJECT_TRACKING_ARTIFACTS_DIR,
                output_path=self.object_tracking_config.DETECT_DIR)

            logging.info(f"Object tracking artifact: {object_tracking_artifacts}")
            logging.info("Exited the initiate_object_tracking method of Object tracking class")
            return object_tracking_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
