import sys
import cv2
import numpy as np
from src.logger import logging
from src.exception import CustomException


def convert_bbox_to_z(bbox):
    try:
        logging.info(
            "Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is the aspect ratio")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    except Exception as e:
        raise CustomException(e, sys) from e


def convert_x_to_bbox(x, score=None):
    logging.info(
        "Takes a bounding box in the centre form [x,y,s,r] and returns it in the form [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right")
    try:
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if (score == None):
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
    except Exception as e:
        raise CustomException(e, sys) from e


def iou_batch(bb_test, bb_gt):
    logging.info("From SORT: Computes IOU between two boxes in the form [x1,y1,x2,y2]")
    try:
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return (o)
    except Exception as e:
        raise CustomException(e, sys) from e


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    try:
        logging.info("Function to Draw Bounding boxes")
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
            label = str(id) + ":" + names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, [255, 255, 255], 1)
            # cv2.circle(img, data, 6, color,-1)
        return img
    except Exception as e:
        raise CustomException(e, sys) from e