import json
import os
import uuid

from inference import Converter
from soccer import Match

class OutputJson:
    def write_detections(self, file_name, frame_id, detections, match: Match):
        sv_detections = Converter.Detections_to_Supervision(detections)
        closest_player = []

        if match.closest_player is not None:
            print(match.closest_player.detection.points)
            xmin = match.closest_player.detection.points[0][0]
            ymin = match.closest_player.detection.points[0][1]
            xmax = match.closest_player.detection.points[1][0]
            ymax = match.closest_player.detection.points[1][1]
            print(xmin)
            print(ymin)
            print(xmax)
            print(ymax)
            closest_player = [
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax)
            ]

            initial_data = {
                "frame_id": frame_id,
                "ball_xyxy": json.dumps(sv_detections[sv_detections.class_id == 0].xyxy.tolist()),
                "player_xyxy": json.dumps(sv_detections[sv_detections.class_id == 1].xyxy.tolist()),
                "possession_counter": json.dumps(match.possession_counter),
                "possession_team": json.dumps(match.team_possession.name),
                "possession_team_counter": json.dumps(match.team_possession.possession),
                "closest_player": json.dumps(closest_player)
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
