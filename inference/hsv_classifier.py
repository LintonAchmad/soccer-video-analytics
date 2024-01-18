import copy
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from inference.base_classifier import BaseClassifier
from inference.colors import all
from sklearn.cluster import KMeans

class HSVClassifier(BaseClassifier):
    def __init__(self, filters: List[dict]):
        """
        Initialize HSV Classifier

        Parameters
        ----------
        filters: List[dict]
            List of colors to classify

            Format:
            [
                {
                    "name": "Boca Juniors",
                    "colors": [inferece.colors.blue, inference.colors.yellow],
                },
                {
                    "name": "River Plate",
                    "colors": [inference.colors.red, inference.colors.white],
                },
                {
                    "name": "Real Madrid",
                    "colors": [inference.colors.white],
                },
                {
                    "name": "Barcelona",
                    "colors": [custom_color],
                },
            ]

            If you want to add a specific color, you can add it as a Python dictionary with the following format:

            custom_color = {
                "name":"my_custom_color",
                "lower_hsv": (0, 0, 0),
                "upper_hsv": (179, 255, 255)
            }

            You can find your custom hsv range with an online tool like https://github.com/hariangr/HsvRangeTool
        """
        super().__init__()

        self.filters = [self.check_filter_format(filter) for filter in filters]

    def check_tuple_format(self, a_tuple: tuple, name: str) -> tuple:
        """
        Check tuple format

        Parameters
        ----------
        a_tuple : tuple
            Tuple to check
        name : str
            Name of the tuple

        Returns
        -------
        tuple
            Tuple checked

        Raises
        ------
        ValueError
            If tuple is not a tuple
        ValueError
            If tuple is not a tuple of 3 elements
        ValueError
            If tuple elements are not integers
        """
        # Check class is a tuple
        if type(a_tuple) != tuple:
            raise ValueError(f"{name} must be a tuple")

        # Check length 3
        if len(a_tuple) != 3:
            raise ValueError(f"{name} must be a tuple of length 3")

        # Check all values are ints
        for value in a_tuple:
            if type(value) != int:
                raise ValueError(f"{name} values must be ints")

    def check_tuple_intervals(self, a_tuple: tuple, name: str):
        """
        Check tuple intervals

        Parameters
        ----------
        a_tuple : tuple
            Tuple to check
        name : str
            Name of the tuple

        Raises
        ------
        ValueError
            If first element is not between 0 and 179
        ValueError
            If second element is not between 0 and 255
        ValueError
            If third element is not between 0 and 255
        """

        # check hue is between 0 and 179
        if a_tuple[0] < 0 or a_tuple[0] > 179:
            raise ValueError(f"{name} hue must be between 0 and 179")

        # check saturation is between 0 and 255
        if a_tuple[1] < 0 or a_tuple[1] > 255:
            raise ValueError(f"{name} saturation must be between 0 and 255")

        # check value is between 0 and 255
        if a_tuple[2] < 0 or a_tuple[2] > 255:
            raise ValueError(f"{name} value must be between 0 and 255")

    def check_color_format(self, color: dict) -> dict:
        """
        Check color format

        Parameters
        ----------
        color : dict
            Color to check

        Returns
        -------
        dict
            Color checked

        Raises
        ------
        ValueError
            If color is not a dict
        ValueError
            If color does not have a name
        ValueError
            If color name is not a string
        ValueError
            If color does not have a lower hsv
        ValueError
            If color does not have an upper hsv
        ValueError
            If lower hsv doesnt have correct tuple format
        ValueError
            If upper hsv doesnt have correct tuple format
        """

        if type(color) != dict:
            raise ValueError("Color must be a dict")
        if "name" not in color:
            raise ValueError("Color must have a name")
        if type(color["name"]) != str:
            raise ValueError("Color name must be a string")
        if "lower_hsv" not in color:
            raise ValueError("Color must have a lower hsv")
        if "upper_hsv" not in color:
            raise ValueError("Color must have an upper hsv")

        self.check_tuple_format(color["lower_hsv"], "lower_hsv")
        self.check_tuple_format(color["upper_hsv"], "upper_hsv")

        self.check_tuple_intervals(color["lower_hsv"], "lower_hsv")
        self.check_tuple_intervals(color["upper_hsv"], "upper_hsv")

        return color

    def check_filter_format(self, filter: dict) -> dict:
        """
        Check filter format

        Parameters
        ----------
        filter : dict
            Filter to check

        Returns
        -------
        dict
            Filter checked

        Raises
        ------
        ValueError
            If filter is not a dict
        ValueError
            If filter does not have a name
        ValueError
            If filter does not have colors
        ValueError
            If filter colors is not a list or a tuple
        """

        if type(filter) != dict:
            raise ValueError("Filter must be a dict")
        if "name" not in filter:
            raise ValueError("Filter must have a name")
        if "colors" not in filter:
            raise ValueError("Filter must have colors")

        if type(filter["name"]) != str:
            raise ValueError("Filter name must be a string")

        if type(filter["colors"]) != list and type(filter["colors"] != tuple):
            raise ValueError("Filter colors must be a list or tuple")

        filter["colors"] = [
            self.check_color_format(color) for color in filter["colors"]
        ]

        return filter

    def get_hsv_img(self, img: np.ndarray) -> np.ndarray:
        """
        Get HSV image

        Parameters
        ----------
        img : np.ndarray
            Image to convert

        Returns
        -------
        np.ndarray
            HSV image
        """
        return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    def apply_filter(self, img: np.ndarray, filter: dict) -> np.ndarray:
        """
        Apply filter to image

        Parameters
        ----------
        img : np.ndarray
            Image to apply filter to
        filter : dict
            Filter to apply

        Returns
        -------
        np.ndarray
            Filtered image
        """
        img_hsv = self.get_hsv_img(img)
        mask = cv2.inRange(img_hsv, filter["lower_hsv"], filter["upper_hsv"])
        return cv2.bitwise_and(img, img, mask=mask)

    def crop_img_for_jersey(self, img: np.ndarray) -> np.ndarray:
        """
        Crop image to get only the jersey part

        Parameters
        ----------
        img : np.ndarray
            Image to crop

        Returns
        -------
        np.ndarray
            Cropped image
        """
        height, width, _ = img.shape

        y_start = int(height * 0.15)
        y_end = int(height * 0.50)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)

        return img[y_start:y_end, x_start:x_end]

    def add_median_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Add median blur to image

        Parameters
        ----------
        img : np.ndarray
            Image to add blur to

        Returns
        -------
        np.ndarray
            Blurred image
        """
        return cv2.medianBlur(img, 5)

    def non_black_pixels_count(self, img: np.ndarray) -> float:
        """
        Returns the amount of non black pixels an image has

        Parameters
        ----------
        img : np.ndarray
            Image

        Returns
        -------
        float
            Count of non black pixels in img
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(img)

    def crop_filter_and_blur_img(self, img: np.ndarray, filter: dict) -> np.ndarray:
        """
        Crops image to get only the jersey part. Filters the colors and adds a median blur.

        Parameters
        ----------
        img : np.ndarray
            Image to crop
        filter : dict
            Filter to apply

        Returns
        -------
        np.ndarray
            Cropped image
        """
        transformed_img = img.copy()
        transformed_img = self.crop_img_for_jersey(transformed_img)
        transformed_img = self.apply_filter(transformed_img, filter)
        transformed_img = self.add_median_blur(transformed_img)
        return transformed_img

    def add_non_black_pixels_count_in_filter(
        self, img: np.ndarray, filter: dict
    ) -> dict:
        """
        Applies filter to image and saves the number of non black pixels in the filter.

        Parameters
        ----------
        img : np.ndarray
            Image to apply filter to
        filter : dict
            Filter to apply to img

        Returns
        -------
        dict
            Filter with non black pixels count
        """
        transformed_img = self.crop_filter_and_blur_img(img, filter)
        filter["non_black_pixels_count"] = self.non_black_pixels_count(transformed_img)
        return filter

    def predict_img(self, img: np.ndarray) -> str:
        """
        Gets the filter with most non blakc pixels on img and returns its name.

        Parameters
        ----------
        img : np.ndarray
            Image to predict

        Returns
        -------
        str
            Name of the filter with most non black pixels on img
        """
        if img is None:
            raise ValueError("Image can't be None")

        filters = copy.deepcopy(self.filters)

        for i, filter in enumerate(filters):
            for color in filter["colors"]:
                color = self.add_non_black_pixels_count_in_filter(img, color)
                if "non_black_pixels_count" not in filter:
                    filter["non_black_pixels_count"] = 0
                filter["non_black_pixels_count"] += color["non_black_pixels_count"]

        max_non_black_pixels_filter = max(
            filters, key=lambda x: x["non_black_pixels_count"]
        )

        return max_non_black_pixels_filter["name"]

    def predict(self, input_image: List[np.ndarray]) -> str:
        """
        Predicts the name of the team from the input image.

        Parameters
        ----------
        input_image : List[np.ndarray]
            Image to predict

        Returns
        -------
        str
            Predicted team name
        """

        if type(input_image) != list:
            input_image = [input_image]

        return [self.predict_img(img) for img in input_image]

    def transform_image_for_every_color(
        self, img: np.ndarray, colors: List[dict] = None
    ) -> List[dict]:
        """
        Transforms image for every color in every filter.

        Parameters
        ----------
        img : np.ndarray
            Image to transform
        colors : List[dict], optional
            List of colors to transform image for, by default None

        Returns
        -------
        List[dict]
            List of Transformed images

            [
                {
                    "red": image,
                },
                {
                    "blue": image,
                }
            ]
        """
        transformed_imgs = {}

        colors_to_transform = all
        if colors:
            colors_to_transform = colors

        for color in colors_to_transform:
            transformed_imgs[color["name"]] = self.crop_filter_and_blur_img(img, color)
        return transformed_imgs

    def plot_every_color_output(
        self, img: np.ndarray, colors: List[dict] = None, save_img_path: str = None
    ):
        """
        Plots every color output of the image.

        Parameters
        ----------
        img : np.ndarray
            Image to plot
        colors : List[dict], optional
            List of colors to plot, by default None
        save_img_path : str, optional
            Path to save image to, by default None
        """
        transformed_imgs = self.transform_image_for_every_color(img, colors)
        transformed_imgs["original"] = img

        n = len(transformed_imgs)

        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))

        fig.suptitle("Every color output")
        for i, (key, value) in enumerate(transformed_imgs.items()):
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
            axs[i].imshow(value)
            if key == "original":
                axs[i].set_title(f"{key}")
            else:
                gray_img = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
                power = cv2.countNonZero(gray_img)
                axs[i].set_title(f"{key}: {power}")
        plt.show()

        if save_img_path is not None:
            fig.savefig(save_img_path)

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
