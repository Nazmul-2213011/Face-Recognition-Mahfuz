import cv2
import os

# Create a folder to store student images
if not os.path.exists("student_images"):
    os.makedirs("student_images")

# Initialize the camera
camera = cv2.VideoCapture(0)

print("Press 'q' to quit at any time.")

while True:
    # Ask for the student's name
    name = input("Enter the student's name (or type 'exit' to quit): ").strip()
    if name.lower() == 'exit':
        print("Exiting...")
        break

    # Create a folder for the student if it doesn't already exist
    student_folder = f"student_images/{name}"
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    # Ask for the number of images to capture
    try:
        image_count = int(input(f"How many images do you want to capture for {name}? "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue

    print(f"Capturing {image_count} images for {name}. Look at the camera...")

    # Capture the specified number of images
    count = 0
    while count < image_count:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Display the frame
        cv2.imshow(f"Capturing Images for {name}", frame)

        # Save the image
        file_path = f"{student_folder}/{name}_{count + 1}.jpg"
        cv2.imwrite(file_path, frame)
        count += 1
        print(f"Captured image {count}/{image_count}")

        # Wait for a short time before capturing the next image
        cv2.waitKey(200)  # 200ms delay between captures

    print(f"Finished capturing {image_count} images for {name}.")

    # Close the window for the current student
    cv2.destroyAllWindows()

# Release the camera and close any remaining windows
camera.release()
cv2.destroyAllWindows()
