import cv2
import face_recognition
import os
import pickle

# Path to the folder containing student images
image_folder = "student_images"
face_encodings = {}

# Define supported image formats
supported_formats = ['.jpg', '.jpeg', '.png']

# Initialize counters for detected and undetected faces
detected_faces_count = 0
undetected_faces_count = 0

# Function to detect faces using OpenCV with more parameters for large or side faces
def detect_faces_opencv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adjust the parameters for better face detection for large or moved faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), maxSize=(1000, 1000))
    
    return faces  # Returns bounding boxes of faces

# Load OpenCV's Haar Cascade face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Starting face detection and encoding...")

# Process images for face encoding
for student_name in os.listdir(image_folder):
    student_path = os.path.join(image_folder, student_name)
    
    if not os.path.isdir(student_path):
        continue  # Skip if not a folder

    print(f"Processing images for {student_name}...")
    encodings = []

    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)

        # Check file format
        if not any(image_name.lower().endswith(fmt) for fmt in supported_formats):
            print(f"Skipping unsupported file '{image_name}'")
            continue

        try:
            # Load image with OpenCV
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Could not read '{image_name}'. Skipping...")
                continue

            # Detect faces using OpenCV (more flexibility with face size and position)
            faces = detect_faces_opencv(frame)
            if len(faces) == 0:
                print(f"No face detected in '{image_name}'. Skipping...")
                undetected_faces_count += 1  # Increment count for undetected faces
                continue

            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection using face_recognition (for encoding)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings_list = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings_list:
                encodings.append(face_encodings_list[0])
                print(f"Encoded face from '{image_name}'")
                detected_faces_count += 1  # Increment count for detected faces
            else:
                print(f"Face detected but encoding failed for '{image_name}'.")
                undetected_faces_count += 1  # Increment count for undetected faces

        except Exception as e:
            print(f"Error processing '{image_name}': {e}")
            undetected_faces_count += 1  # Increment count for undetected faces

    if encodings:
        face_encodings[student_name] = encodings
    else:
        print(f"No valid encodings found for {student_name}. Skipping student.")

# Save encodings to file
encoding_file = "face_encodings.pkl"
try:
    with open(encoding_file, "wb") as f:
        pickle.dump(face_encodings, f)
    print(f"Face encoding completed and saved to '{encoding_file}'")
except Exception as e:
    print(f"Error saving encodings: {e}")

# Print the results
print(f"\nTotal images with face detected: {detected_faces_count}")
print(f"Total images with face NOT detected: {undetected_faces_count}")
