import argparse

import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

from inference import Converter, HSVClassifier, InertiaClassifier, YoloV8, OutputJson
from inference.filters import filters
from run_utils import (
    get_main_ball,
    get_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/3.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)

# Object Detectors
detector = YoloV8()

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)

# Add inertia to classifier
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
chelsea = Team(
    name="Player-R",
    abbreviation="PLR",
    color=(255, 0, 0),
    board_color=(244, 86, 64),
    text_color=(255, 255, 255),
)
man_city = Team(name="Player-L", abbreviation="PLL", color=(240, 230, 188))
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city

# Tracking
player_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=250,
    initialization_delay=3,
    hit_counter_max=90,
)

ball_tracker = Tracker(
    distance_function=mean_euclidean,
    distance_threshold=150,
    initialization_delay=20,
    hit_counter_max=2000,
)
motion_estimator = MotionEstimator()
coord_transformations = None

# Paths
path = AbsolutePath()
converter = Converter()

# Get Counter img
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

ouput_json = OutputJson()
json_name = ouput_json.generate_random_filename()

src=cv2.imread('src.jpg')
dst=cv2.imread('dst.jpg')
coptemp=dst.copy()

pts1 = np.float32([[377,176],[1368,169],[1754,722],[4,710]])
pts2 = np.float32([[450,33],[540,300],[362,302],[450,567]])

M = cv2.getPerspectiveTransform(pts1,pts2)

kits_clf = None
left_team_label = 0
grass_hsv = None

height = int(video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

for i, frame in enumerate(video):

    # Get Detections
    annotated_frame = cv2.resize(frame, (width, height))
    df = get_detections(detector, annotated_frame)

    # Get the players boxes and kit colors
    players_imgs, players_boxes = hsv_classifier.get_players_boxes(result=df)
    kits_colors = hsv_classifier.get_kits_colors(players_imgs, grass_hsv, annotated_frame)

    if i == 0:
        kits_clf = hsv_classifier.get_kits_classifier(kits_colors)
        left_team_label = hsv_classifier.get_left_team_label(players_boxes, kits_colors, kits_clf)
        grass_color = hsv_classifier.get_grass_color(df.orig_img)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    players_detections = converter.DataFrame_to_Detections(
        df.boxes[df.boxes.cls == 0],
        width=width,
        left_team_label=left_team_label,
        kits_clf=kits_clf,
        orig_img=df.orig_img,
        grass_hsv=grass_hsv
    )

    ball_detections = converter.DataFrame_to_Detections(
        df.boxes[df.boxes.cls == 3],
        width=width,
        left_team_label=left_team_label,
        kits_clf=kits_clf,
        orig_img=df.orig_img,
        grass_hsv=grass_hsv
    )
    detections = ball_detections + players_detections

    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=detections,
        frame=frame,
    )

    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )

    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )

    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)

    player_detections = classifier.predict_from_detections(
        detections=player_detections,
        img=frame,
    )

    # Match update
    ball = get_main_ball(ball_detections)
    players = Player.from_detections(detections=player_detections, teams=teams)

    match.update(players, ball)

    # Draw
    frame = PIL.Image.fromarray(frame)

    if args.possession:
        frame = Player.draw_players(
            players=players, frame=frame, confidence=False, id=True
        )

        frame = path.draw(
            img=frame,
            detection=ball.detection,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )

        frame = match.draw_possession_counter(
            frame, counter_background=possession_background, debug=False
        )

        if ball:
            frame = ball.draw(frame)

    if args.passes:
        pass_list = match.passes

        frame = Pass.draw_pass_list(
            img=frame, passes=pass_list, coord_transformations=coord_transformations
        )

        frame = match.draw_passes_counter(
            frame, counter_background=passes_background, debug=False
        )

    frame = np.array(frame)

    # Write to json output file LFG
    ouput_json.write_detections(json_name, i, df, match=match, matrix=M)

    # Write video
    video.write(frame)
