import json
import os
import uuid
import cv2
import numpy as np

from inference import Converter
from soccer import Match

class OutputJson:
    def write_detections(self, file_name, frame_id, detections, match: Match, matrix):     
        """
        Write detections to json file

        ----------
        possession_counter: counts how long team has possession for
        """

        sv_detections = Converter.Detections_to_Supervision(detections)
        closest_player = self.get_closest_player_bbox(match)
        game_possession = match.get_possession()
        team_passes = match.get_team_passes()
        match_passes = match.get_match_passes()
        pitchMap = self.get_pitch_map(sv_detections[sv_detections.class_id == 1].xyxy, matrix)

        initial_data = {
            "frame_id": frame_id,
            "ball_xyxy": json.dumps(sv_detections[sv_detections.class_id == 0].xyxy.tolist()),
            "player_xyxy": json.dumps(sv_detections[sv_detections.class_id == 1].xyxy.tolist()),
            "possession_counter": json.dumps(match.possession_counter),
            "possession_team": json.dumps(match.team_possession.name),
            "possession_team_counter": json.dumps(match.team_possession.possession),
            "closest_player": json.dumps(closest_player),
            "game_possession": json.dumps(game_possession),
            "team_passes": json.dumps(team_passes),
            "match_passes": json.dumps(match_passes),
            "pitch_map": json.dumps(pitchMap),
        }

        self.append_to_json(file_name, initial_data) 

    def generate_random_filename(self):
        random_name = str(uuid.uuid4())  # Generates a random UUID
        return f"{random_name}.json"

    def append_to_json(self, file_name, new_data):
        if not os.path.exists(file_name):
            # Create file with initial data as a list
            with open(file_name, 'w') as file:
                json.dump([new_data], file, indent=4)
        else:
            # Read existing data
            with open(file_name, 'r') as file:
                data = json.load(file)

            # Append new data
            data.append(new_data)

            # Write back to file
            with open(file_name, 'w') as file:
                json.dump(data, file, indent=4)

    def get_closest_player_bbox(self, match: Match):
        if match.closest_player is not None:
            xmin = match.closest_player.detection.points[0][0]
            ymin = match.closest_player.detection.points[0][1]
            xmax = match.closest_player.detection.points[1][0]
            ymax = match.closest_player.detection.points[1][1]

            closest_player = [
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax)
            ]

            return closest_player
        else:
            return []

    def get_pitch_map(self, players, matrix):
        points = []

        for p in players:
            tl = np.float32([[p[0], p[1]]])
            br = np.float32([[p[2], p[3]]])

            # Apply perspective transformation
            tl_transformed = cv2.perspectiveTransform(tl[None, :, :], matrix)
            br_transformed = cv2.perspectiveTransform(br[None, :, :], matrix)

            # Extract transformed coordinates
            x1, y1 = int(tl_transformed[0][0][0]), int(tl_transformed[0][0][1])
            x2, y2 = int(br_transformed[0][0][0]), int(br_transformed[0][0][1])
            points.append((x1, y1))

        return points
