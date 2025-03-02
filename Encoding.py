import face_recognition
import os
import pickle
from PIL import Image

# Path to the folder containing student images
image_folder = "student_images"

# Dictionary to store encodings and names
face_encodings = {}
print("Encoding faces...")

# Define supported image formats
supported_formats = ['.jpg', '.jpeg', '.png']

# Loop through each student's folder
for student_name in os.listdir(image_folder):
    student_path = os.path.join(image_folder, student_name)
    if not os.path.isdir(student_path):
        continue

    print(f"Processing images for {student_name}...")
    encodings = []

    # Loop through each image in the student's folder
    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)

        # Check if the image format is supported
        if not any(image_name.lower().endswith(fmt) for fmt in supported_formats):
            print(f"Warning: Unsupported format for {image_name}. Skipping...")
            continue

        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)

            # Detect and encode the face
            face_locations = face_recognition.face_locations(image)
            face_encoding = face_recognition.face_encodings(image, face_locations)

            # If a face is found, add the encoding
            if face_encoding:
                encodings.append(face_encoding[0])
            else:
                print(f"Warning: No face detected in {image_name}. Skipping...")

        except Exception as e:
            print(f"Error processing {image_name}: {e}. Skipping...")

    # Store the encodings for the student
    if encodings:
        face_encodings[student_name] = encodings
    else:
        print(f"Warning: No valid encodings found for {student_name}.")

# Save the encodings to a file for later use
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(face_encodings, f)

print("Face encoding completed and saved to 'face_encodings.pkl'")
