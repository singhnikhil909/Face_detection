import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch

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

# Load and encode the target person's image
target_image_path = "/Users/nikhilsingh/Desktop/face_1/data/face_1.jpg"
target_image = cv2.imread(target_image_path)
rgb_target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
detection_result = detector.detect_faces(rgb_target_image)
if len(detection_result) == 0:
    raise ValueError("No faces found in the target image!")
x, y, width, height = detection_result[0]['box']
target_face = rgb_target_image[y:y+height, x:x+width]
target_embedding = get_face_embedding(facenet, target_face)

# Load the video
video_path = "/Users/nikhilsingh/Desktop/face_1/data/MVI_3687.MP4"
video_capture = cv2.VideoCapture(video_path)

# Create an output video writer to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = "output_video.mp4"
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
frame_skip = 5  # Process every 5th frame

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    detection_results = detector.detect_faces(rgb_frame)

    for result in detection_results:
        x, y, width, height = result['box']
        face = rgb_frame[y:y+height, x:x+width]
        face_embedding = get_face_embedding(facenet, face)

        # Compare the detected face with the target face
        distance = np.linalg.norm(target_embedding - face_embedding)
        matching_score = 1 / (1 + distance)

        if matching_score > 0.6:  # Adjust this threshold as needed
            # Draw a bounding box around the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Label the face with the person's name and matching score
            label = f"Person: {matching_score:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

# Release video capture and writer objects
video_capture.release()
out.release()

print("Processing complete. Annotated video saved as", output_video_path)

