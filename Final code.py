import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import uuid

def process_video(video_path, output_dir="output"):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load a pretrained YOLOv8n model
    yolo_model = YOLO('model/epoch 100/best_e100.pt')
    font = ImageFont.truetype("arial.ttf", 40)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file_name = os.path.join(output_dir, f"{uuid.uuid4().hex}.mp4")
    out = cv2.VideoWriter(output_file_name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

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
                draw.rectangle([x1, y1, x2, y2], outline="Red", width=3)

        # Write the count of boxes detected on the frame
        draw.text((10, 10), f"Empty Spaces: {frame_boxes}", fill=(236, 15, 25), font=font)

        # Convert PIL image back to OpenCV format
        frame_with_boxes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Write the frame with bounding boxes
        out.write(frame_with_boxes)

        # Show the video processing in real-time
        cv2.imshow('Video Processing', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Processed {frame_count} frames.")

    # Release everything if the job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        return None
    return output_file_name  # return the path to the saved video

# To use the function
output_path = process_video('videos/Ayush.mp4')
if output_path:
    print(f"Processed video saved at: {output_path}")
else:
    print("No frames were processed.")