# UTILITIES FOR THE CAR DETECTION AND LICENSE PLATA RECOGNITION:

import torch
import cv2
import os


# Prediction and Detection Function:
# Function to predict and annotate detected objects (e.g., license plates) on an image
# using a YOLO model. Outputs the image with bounding boxes and confidence levels.

def predict_and_detect_v1(model, img, conf=0.5, rectangle_thickness=2, text_thickness=2):
    """
    Predict and annotate objects in an image using a YOLO model.

    Parameters:
        model (YOLO): Pre-trained YOLO model.
        img (ndarray): Input image for detection.
        conf (float): Confidence threshold for predictions. Default is 0.5.
        rectangle_thickness (int): Thickness of bounding box rectangles. Default is 2.
        text_thickness (int): Thickness of confidence level text. Default is 2.

    Returns:
        img (ndarray): Annotated image with bounding boxes and confidence labels.
    """
    # Make Predictions
    # Use the YOLO model to predict objects in the input image with the specified confidence threshold
    results = model.predict(source=img, conf=conf)

    # Iterate Over Results
    for result in results:
        # Iterate through each detected bounding box in the result
        for box in result.boxes:
            # Extract the coordinates of the top-left (x1, y1) and bottom-right (x2, y2) of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers

            # Draw Bounding Box
            # Draw a rectangle on the image to highlight the detected object
            # (0, 0, 255) specifies the color as red in BGR format
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), rectangle_thickness)

            # Add Confidence Label
            # Format the confidence level of the detection as text
            text = f"License Plate: {box.conf[0].item():.2f}"

            # Place the text just above the bounding box
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), text_thickness)

    # Return Annotated Image
    # Return the input image with bounding boxes and confidence labels added
    return img

# ---------------------------------------------------------------------------------------------------------

# Create Video Writer Function
# Function to initialize a video writer object for saving processed video frames.

def create_video_writer(video_cap, output_filename):
    """
    Create a video writer for saving processed video frames.

    Parameters:
        video_cap (cv2.VideoCapture): OpenCV video capture object for the input video.
        output_filename (str): Path to save the output video.

    Returns:
        cv2.VideoWriter: Video writer object configured for the input video properties.
    """
    # Retrieve Video Properties
    # Get the width and height of the video frames
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the frames per second (fps) of the input video
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # Define Video Codec
    # Define the codec for the output video ('mp4v' for MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriter Object
    # Create the VideoWriter object to write the output video with the correct properties
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Return Video Writer
    # Return the VideoWriter object for saving video frames
    return writer

# ---------------------------------------------------------------------------------------------------------

# Process Video for Object Detection
# This function processes a video file, detects objects in each frame using a YOLO model,
# and saves the processed video with bounding boxes and annotations.

def process_video(input_video_path, output_filename, model, conf=0.5):
    """
    Process a video file for object detection using a YOLO model and save the processed video.

    Parameters:
        input_video_path (str): Path to the input video file.
        output_filename (str): Name of the output video file.
        model (YOLO): Trained YOLO model for object detection.
        conf (float): Confidence threshold for object detection (default: 0.5).
    """
    # Setup Output Directory
    # Define the output directory for processed videos
    output_dir = "/content/outputs"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Generate the full path for the output video file
    output_video_path = os.path.join(output_dir, output_filename)

    # Open Input Video
    # Use OpenCV to open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open file.")  # Print an error message if the video can't be opened
        return

    # Initialize Video Writer
    # Create a VideoWriter object for saving the processed video
    writer = create_video_writer(cap, output_video_path)

    # Initialize a frame counter
    frame_count = 0

    # Process Each Frame
    while True:
        # Read a frame from the video
        success, img = cap.read()

        # Break the loop if the frame can't be read (end of video or error)
        if not success:
            print("Error: Unable to read frame or video has ended.")
            break

        # Detect objects in the current frame and annotate it
        result_img = predict_and_detect(model, img, conf=conf)

        # Write the processed frame to the output video
        writer.write(result_img)
        cv2.waitKey(1)  # Wait 1 millisecond between frames (adjustable)

        # Increment the frame counter and print progress
        frame_count += 1
        print(f"Frame {frame_count} processed.")

    # Release Resources
    # Release the VideoCapture and VideoWriter resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Final Message
    print(f"The video has been processed successfully and saved to {output_video_path}.")

# ---------------------------------------------------------------------------------------------------------

# Convert the region of interest (ROI) from BGR to HSV (Hue, Saturation, Value).
# HSV is better for color detection compared to BGR because it separates the color (Hue) from the intensity (Value).

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Define the color ranges in HSV. Each color has a "lower" and "upper" bound that represents the range of hues, saturation, and value.
    colors = {
        "Red": ([0, 120, 70], [10, 255, 255]),  # Red color range
        "Green": ([36, 25, 25], [70, 255, 255]),  # Green color range
        "Blue": ([94, 80, 2], [126, 255, 255]),  # Blue color range
        "Yellow": ([15, 150, 150], [35, 255, 255]),  # Yellow color range
        "Black": ([0, 0, 0], [180, 255, 30]),  # Black color range
        "White": ([0, 0, 200], [180, 20, 255]),  # White color range
        "Gray": ([0, 0, 40], [180, 20, 200])  # Gray color range
    }

    max_pixels = 0  # Initialize a variable to track the maximum number of pixels detected for any color.
    dominant_color = "Unknown"  # Set the default dominant color to "Unknown".

    # Loop through the colors dictionary to apply each color mask and detect pixels within the defined color range.
    for color, (lower, upper) in colors.items():
        # Apply the color mask for the current color, where "lower" and "upper" define the HSV range for that color.
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Count the number of non-zero pixels (i.e., pixels that match the color in the mask).
        num_pixels = cv2.countNonZero(mask)

        # If the number of matching pixels is greater than the previously found maximum, update the dominant color.
        if num_pixels > max_pixels:
            max_pixels = num_pixels  # Update the maximum number of pixels detected for the current color.
            dominant_color = color  # Set the dominant color to the current color.

    # Return the dominant color based on the highest number of matching pixels found in the ROI.
    return dominant_color

# ---------------------------------------------------------------------------------------------------------
# Updated predict_and_detect to use EasyOCR and color detection:

def predict_and_detect(model, img, conf=0.5, rectangle_thickness=2, text_thickness=2):
    """
    Predict objects in an image using the YOLO model and perform OCR and color detection.
    
    Args:
    - model: The YOLO model used for object detection.
    - img: The input image to process.
    - conf: Confidence threshold for YOLO predictions (default=0.5).
    - rectangle_thickness: Thickness of the bounding box rectangle (default=2).
    - text_thickness: Thickness of the text overlay (default=2).

    Returns:
    - img: The processed image with detections, OCR results, and color annotations.
    """
    # Use YOLO to predict objects in the image with a specified confidence threshold.
    results = model.predict(source=img, conf=conf)

    # Iterate through the prediction results.
    for result in results:
        # Process each detected bounding box in the current frame/image.
        for box in result.boxes:
            # Extract bounding box coordinates and convert to integers.
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extract the region of interest (ROI) containing the detected license plate.
            plate_img = img[y1:y2, x1:x2]

            # Expand the bounding box region to include additional vehicle context for color detection.
            expand_y = int((y2 - y1) * 4)  # Vertical margin expansion factor.
            expand_x = int((x2 - x1) * 3)  # Horizontal margin expansion factor.

            # Adjust the expanded bounding box coordinates, ensuring they stay within the image boundaries.
            y1_expanded = max(y1 - expand_y // 2, 0)
            y2_expanded = min(y2 + expand_y // 2, img.shape[0])
            x1_expanded = max(x1 - expand_x // 2, 0)
            x2_expanded = min(x2 + expand_x // 2, img.shape[1])

            # Extract the expanded region of interest (ROI) for vehicle color detection.
            car_roi = img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            # Use EasyOCR to read text from the extracted license plate image.
            ocr_result = reader.readtext(plate_img)

            # Initialize license plate text as "Unknown" in case no text is detected.
            license_plate_text = "Unknown"

            # If OCR detects text, extract the license plate number.
            if ocr_result:
                license_plate_text = ocr_result[0][-2]  # Extract the detected text string.

            # Detect the dominant color in the expanded vehicle ROI.
            car_color = detect_color(car_roi)

            # Draw a red rectangle around the detected license plate in the original image.
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), rectangle_thickness)

            # Annotate the image with the detected license plate text and vehicle color.
            text = f"{license_plate_text}, {car_color}"  # Combine the text for annotation.
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), text_thickness)

    # Return the processed image with added detections and annotations.
    return img
