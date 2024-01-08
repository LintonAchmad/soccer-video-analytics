import json
import os
import uuid

class OutputJson:
    def __init__(self, file_name, frame_id):
        initial_data = {
                "frame_id": frame_id,
                "xyxy": json.dumps(detections[detections.class_id == 0].xyxy.tolist()),
                "confidence": json.dumps(detections[detections.class_id == 0].confidence.tolist()),
                "class_id": json.dumps(detections[detections.class_id == 0].class_id.tolist()),
                "possession": json.dumps(player_in_possession_detection)
            }

        self.append_to_json(file_name, initial_data)

    def generate_random_filename():
        random_name = str(uuid.uuid4())  # Generates a random UUID
        return f"{random_name}.json"

    def append_to_json(file_name, new_data):
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
