from typing import List

import numpy as np
import supervision as sv
import torch
import sys

from inference.base_detector import BaseDetector
from ultralytics import YOLO

class YoloV8(BaseDetector):
    def __init__(
        self
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will download the model with COCO weights
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO("models/last.pt")

    def predict(self, input_image: List[np.ndarray]):
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        result = self.model(input_image, verbose=False, conf=0.35, iou=0.7)[0]
        # print(result.boxes)
        # detections = sv.Detections.from_ultralytics(result)
        # print(detections)

        return result
