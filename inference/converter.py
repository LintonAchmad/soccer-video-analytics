from typing import List

import cv2
import norfair
import numpy as np
import pandas as pd
import supervision as sv
from sklearn.cluster import KMeans
class Converter:
    def __init__(
        self
    ):
        self.labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]

    def DataFrame_to_Detections(self, df: List, width, left_team_label, kits_clf, orig_img, grass_hsv) -> List[norfair.Detection]:
        """
        Converts a DataFrame to a list of norfair.Detection

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the bounding boxes

        Returns
        -------
        List[norfair.Detection]
            List of norfair.Detection
        """

        detections = []

        for row in df:
            # get the bounding box coordinates
            xmin = row.xyxy[0][0]
            ymin = row.xyxy[0][1]
            xmax = row.xyxy[0][2]
            ymax = row.xyxy[0][3]

            box = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax],
                ]
            )

            confidence = row.conf[0]

            label = int(row.cls.numpy()[0])
            x1, y1, x2, y2 = map(int, row.xyxy[0].numpy())

            # If the box contains a player, find to which team he belongs
            if label == 0:
                kit_color = self.get_kits_colors([orig_img[y1: y2, x1: x2]], grass_hsv)
                team = self.classify_kits(kits_clf, kit_color)
                if team == left_team_label:
                    label = 0
                else:
                    label = 1

            # If the box contains a Goalkeeper, find to which team he belongs
            elif label == 1:
                if x1 < 0.5 * width:
                    label = 2
                else:
                    label = 3

            # Increase the label by 2 because of the two add labels "Player-L", "GK-L"
            else:
                label = label + 2

            # get the predicted class
            if row.cls[0] == 0:
                name = "player"

                data = {
                    "name": name,
                    "p": confidence,
                    "label": self.labels[label],
                    "kit_color": kit_color,
                    "team": team,
                }
            else:
                name = "ball"
                data = {
                    "name": name,
                    "p": confidence,
                }

            detection = norfair.Detection(
                points=box,
                data=data,
            )

            detections.append(detection)

        return detections

    @staticmethod
    def Detections_to_DataFrame(detections: List[norfair.Detection]) -> pd.DataFrame:
        """
        Converts a list of norfair.Detection to a DataFrame

        Parameters
        ----------
        detections : List[norfair.Detection]
            List of norfair.Detection

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        df = pd.DataFrame()

        for detection in detections:

            xmin = detection.points[0][0]
            ymin = detection.points[0][1]
            xmax = detection.points[1][0]
            ymax = detection.points[1][1]

            name = detection.data["name"]
            confidence = detection.data["p"]

            data = {
                "xmin": [xmin],
                "ymin": [ymin],
                "xmax": [xmax],
                "ymax": [ymax],
                "name": [name],
                "confidence": [confidence],
            }

            # get color if its in data
            if "color" in detection.data:
                data["color"] = [detection.data["color"]]

            if "label" in detection.data:
                data["label"] = [detection.data["label"]]

            if "classification" in detection.data:
                data["classification"] = [detection.data["classification"]]

            df_new_row = pd.DataFrame.from_records(data)

            df = pd.concat([df, df_new_row])

        return df

    @staticmethod
    def TrackedObjects_to_Detections(
        tracked_objects: List[norfair.tracker.TrackedObject],
    ) -> List[norfair.Detection]:
        """
        Converts a list of norfair.tracker.TrackedObject to a list of norfair.Detection

        Parameters
        ----------
        tracked_objects : List[norfair.tracker.TrackedObject]
            List of norfair.tracker.TrackedObject

        Returns
        -------
        List[norfair.Detection]
            List of norfair.Detection
        """

        live_objects = [
            entity for entity in tracked_objects if entity.live_points.any()
        ]

        detections = []

        for tracked_object in live_objects:
            detection = tracked_object.last_detection
            detection.data["id"] = int(tracked_object.id)
            detections.append(detection)

        return detections

    @staticmethod
    def Detections_to_Supervision(detections) -> pd.DataFrame:
        results = sv.Detections.from_ultralytics(detections)

        return results
    
    def get_grass_color(self, img):
        """
        Finds the color of the grass in the background of the image

        Args:
            img: np.array object of shape (WxHx3) that represents the BGR value of the
            frame pixels .

        Returns:
            grass_color
                Tuple of the BGR value of the grass color in the image
        """
        # Convert image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Calculate the mean value of the pixels that are not masked
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        grass_color = cv2.mean(img, mask=mask)
        return grass_color[:3]

    def get_players_boxes(self, result):
        """
        Finds the images of the players in the frame and their bounding boxes.

        Args:
            result: ultralytics.engine.results.Results object that contains all the
            result of running the object detection algroithm on the frame

        Returns:
            players_imgs
                List of np.array objects that contain the BGR values of the cropped
                parts of the image that contains players.
            players_boxes
                List of ultralytics.engine.results.Boxes objects that contain various
                information about the bounding boxes of the players found in the image.
        """
        players_imgs = []
        players_boxes = []
        for box in result.boxes:
            label = int(box.cls.numpy()[0])
            if label == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                player_img = result.orig_img[y1: y2, x1: x2]
                players_imgs.append(player_img)
                players_boxes.append(box)
        return players_imgs, players_boxes

    def get_kits_colors(self, players, grass_hsv=None, frame=None):
        """
        Finds the kit colors of all the players in the current frame

        Args:
            players: List of np.array objects that contain the BGR values of the image
            portions that contain players.
            grass_hsv: tuple that contain the HSV color value of the grass color of
            the image background.

        Returns:
            kits_colors
                List of np arrays that contain the BGR values of the kits color of all
                the players in the current frame
        """
        kits_colors = []
        if grass_hsv is None:
            grass_color = self.get_grass_color(frame)
            grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

        for player_img in players:
            # Convert image to HSV color space
            hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

            # Define range of green color in HSV
            lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
            upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Bitwise-AND mask and original image
            mask = cv2.bitwise_not(mask)
            upper_mask = np.zeros(player_img.shape[:2], np.uint8)
            upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
            mask = cv2.bitwise_and(mask, upper_mask)

            kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])

            kits_colors.append(kit_color)
        return kits_colors

    def get_kits_classifier(self, kits_colors):
        """
        Creates a K-Means classifier that can classify the kits accroding to their BGR
        values into 2 different clusters each of them represents one of the teams

        Args:
            kits_colors: List of np.array objects that contain the BGR values of
            the colors of the kits of the players found in the current frame.

        Returns:
            kits_kmeans
                sklearn.cluster.KMeans object that can classify the players kits into
                2 teams according to their color..
        """
        kits_kmeans = KMeans(n_clusters=2)
        kits_kmeans.fit(kits_colors)
        return kits_kmeans

    def classify_kits(self, kits_classifer, kits_colors):
        """
        Classifies the player into one of the two teams according to the player's kit
        color

        Args:
            kits_classifer: sklearn.cluster.KMeans object that can classify the
            players kits into 2 teams according to their color.
            kits_colors: List of np.array objects that contain the BGR values of
            the colors of the kits of the players found in the current frame.

        Returns:
            team
                np.array object containing a single integer that carries the player's
                team number (0 or 1)
        """
        team = kits_classifer.predict(kits_colors)
        return team

    def get_left_team_label(self, players_boxes, kits_colors, kits_clf):
        """
        Finds the label of the team that is on the left of the screen

        Args:
            players_boxes: List of ultralytics.engine.results.Boxes objects that
            contain various information about the bounding boxes of the players found
            in the image.
            kits_colors: List of np.array objects that contain the BGR values of
            the colors of the kits of the players found in the current frame.
            kits_clf: sklearn.cluster.KMeans object that can classify the players kits
            into 2 teams according to their color.
        Returns:
            left_team_label
                Int that holds the number of the team that's on the left of the image
                either (0 or 1)
        """
        left_team_label = 0
        team_0 = []
        team_1 = []

        for i in range(len(players_boxes)):
            x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].numpy())

            team = self.classify_kits(kits_clf, [kits_colors[i]]).item()
            if team==0:
                team_0.append(np.array([x1]))
            else:
                team_1.append(np.array([x1]))

        team_0 = np.array(team_0)
        team_1 = np.array(team_1)

        if np.average(team_0) - np.average(team_1) > 0:
            left_team_label = 1

        return left_team_label
