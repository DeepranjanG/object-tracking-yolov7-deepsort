import sys
import numpy as np
from filterpy.kalman import KalmanFilter
from src.utils import convert_bbox_to_z, convert_x_to_bbox
from src.exception import CustomException


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        
        Parameter 'bbox' must have 'detected class' int number at the -1 position.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10. # R: Covariance matrix of measurement noise (set to high for noisy inputs -> more 'inertia' of boxes')
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.5 # Q: Covariance matrix of process noise (set to high for erratically moving things)
        self.kf.Q[4:,4:] *= 0.5

        self.kf.x[:4] = convert_bbox_to_z(bbox) # STATE VECTOR
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centroidarr = []
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        
        #keep yolov5 detected class information
        self.detclass = bbox[5]

        # If we want to store bbox
        self.bbox_history = [bbox]
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        try:
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            self.detclass = bbox[5]
            CX = (bbox[0]+bbox[2])//2
            CY = (bbox[1]+bbox[3])//2
            self.centroidarr.append((CX,CY))
            self.bbox_history.append(bbox)
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        try:
            if((self.kf.x[6]+self.kf.x[2])<=0):
                self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
            if(self.time_since_update>0):
                self.hit_streak = 0
            self.time_since_update += 1
            self.history.append(convert_x_to_bbox(self.kf.x))
            # bbox=self.history[-1]
            # CX = (bbox[0]+bbox[2])/2
            # CY = (bbox[1]+bbox[3])/2
            # self.centroidarr.append((CX,CY))
            
            return self.history[-1]
        except Exception as e:
            raise CustomException(e, sys) from e