import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import os
import json

# Initialize MTCNN detector and FaceNet model
detector = MTCNN()
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to get face embedding
def get_face_embedding(model, face):
    face = cv2.resize(face, (160, 160))
    face = np.transpose(face, (2, 0, 1))
    face = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
    face = (face - 127.5) / 128.0  # Normalize as required by FaceNet
    embedding = model(face)
    return embedding.detach().numpy().flatten()

# Load and encode target images
target_images_directory = "/Users/nikhilsingh/Desktop/face_1/data"
target_images = {}
for filename in os.listdir(target_images_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(target_images_directory, filename)
        target_image = cv2.imread(image_path)
        rgb_target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect_faces(rgb_target_image)
        if len(detection_result) > 0:
            x, y, width, height = detection_result[0]['box']
            target_face = rgb_target_image[y:y+height, x:x+width]
            target_embedding = get_face_embedding(facenet, target_face)
            target_images[filename] = {
                'name': filename.split('.')[0],
                'embedding': target_embedding,
                'path': image_path
            }
            print(f"Loaded target image: {filename} with embedding size: {target_embedding.shape}")
        else:
            print(f"No face detected in target image: {filename}")

# Process each video and detect faces
video_directory ="/Users/nikhilsingh/Desktop/face_1/videos"
results = []

for video_filename in os.listdir(video_directory):
    if video_filename.endswith(".mp4")or  video_filename.endswith(".MP4"):
        video_path = os.path.join(video_directory, video_filename)
        video_capture = cv2.VideoCapture(video_path)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))

        print(f"Processing video: {video_filename}")

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Timestamp in seconds

            # Detect faces in the frame
            detection_results = detector.detect_faces(rgb_frame)
            if not detection_results:
                print(f"No faces detected in frame at {timestamp} seconds in video {video_filename}")

            for result in detection_results:
                x, y, width, height = result['box']
                face = rgb_frame[y:y+height, x:x+width]
                face_embedding = get_face_embedding(facenet, face)

                # Compare the detected face with the target faces
                for target_name, target_info in target_images.items():
                    distance = np.linalg.norm(target_info['embedding'] - face_embedding)
                    matching_score = 1 / (1 + distance)

                    print(f"Comparing face in video {video_filename} at {timestamp} seconds with {target_info['name']} with score {matching_score:.2f}")

                    if matching_score > 0.6:  # Adjust this threshold as needed
                        print(f"Match found for {target_info['name']} in video {video_filename} at {timestamp} seconds with score {matching_score:.2f}")
                        # Record the appearance time
                        result_entry = {
                            'person_name': target_info['name'],
                            'image_directory': target_info['path'],
                            'video': {
                                'directory': video_path,
                                'timestamp': timestamp
                            }
                        }
                        results.append(result_entry)

        video_capture.release()

# Ensure every person has an entry in the JSON, even if they don't appear in any video
final_results = []
for target_name, target_info in target_images.items():
    person_entry = {
        'person_name': target_info['name'],
        'image_directory': target_info['path'],
        'videos': []
    }
    for result in results:
        if result['person_name'] == target_info['name']:
            person_entry['videos'].append(result['video'])
    final_results.append(person_entry)

# Save the results in a JSON file
output_json_path = "appearance_times.json"
with open(output_json_path, 'w') as f:
    json.dump(final_results, f, indent=4)

print(f"Processing complete. Results saved in {output_json_path}")
print(f"Total results: {len(results)}")


