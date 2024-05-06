import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from Email_Varification import *

# Load a pretrained YOLOv8n model
yolo_model = YOLO('model/epoch 100/best_e100.pt')

# Load the video
video_path = 'videos/Ayush.mp4'
cap = cv2.VideoCapture(video_path)
font = ImageFont.truetype("arial.ttf", 40)

# Initialize the count of empty shelves
empty_shelf_count = 0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/3eded.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = yolo_model(image)

    draw = ImageDraw.Draw(image)

    # Count of boxes detected in this frame
    frame_boxes = sum(len(r.boxes.xyxy) for r in results)

    # Annotate each detected box
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Calculate the percentage of empty spaces
    total_pixels = 36
    empty_percentage = (frame_boxes / total_pixels) * 100
    available_percentage = 100 - empty_percentage

    # Define the background box coordinates and size
    box_x1, box_y1, box_x2, box_y2 = 10, 10, 600, 110  # You can adjust the coordinates and size as needed

    # Draw a background box
    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="black")

    # Display the count of empty spaces in the background box
    draw.text((box_x1 + 10, box_y1 + 10), f"Empty Spaces: {frame_boxes}", fill="white", font=font)
    # Display the percentage of empty spaces in the background box below the count
    draw.text((box_x1 + 10, box_y1 + 50), f"Available Product[%]: {available_percentage:.2f}%", fill="white", font=font)

    # Convert PIL image back to OpenCV format
    frame_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Write the frame with bounding boxes
    out.write(frame_with_boxes)

    # Send an email notification when Percentage > 5%
    if available_percentage < 85:
        subject = 'Empty Spaces Detected'
        message = f'The count of empty spaces detected is: {frame_boxes}, Percentage: {empty_percentage:.2f}%'
        stock_notification_sent = True

        # Capture image of the empty shelf
        empty_shelf_image_path = 'Images/empty_shelf.jpg'
        cv2.imwrite(empty_shelf_image_path, frame)

        image_with_outlines = np.array(image)
        cv2.imwrite(empty_shelf_image_path, cv2.cvtColor(image_with_outlines, cv2.COLOR_RGB2BGR))

        empty_shelf_image_captured = True

        # Attach the empty shelf image if captured
        attachments = []
        if empty_shelf_image_captured:
            attachments.append(empty_shelf_image_path)

        # Send email with or without attachments
        send_email(subject, message, attachments)

    # Restock Notification
    if available_percentage > 98:
        subject = 'Restock Alert'
        message = f'Products have been restocked: {empty_percentage:.2f}%'
        send_email(subject, message)
        stock_notification_sent = True

    # Show the video processing in real-time
    cv2.imshow('Video Processings', frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()