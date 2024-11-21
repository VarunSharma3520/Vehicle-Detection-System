import cv2
import numpy as np
import os
# import winsound
import torch
import logging
from PIL import Image
import google.generativeai as genai
from datetime import datetime
import textwrap
from gtts import gTTS
# from playsound import playsound
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure the Gemini API
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
if not GENAI_API_KEY:
    logger.error("Gemini API key is not set. Please set the GENAI_API_KEY environment variable.")
    exit(1)

genai.configure(api_key=GENAI_API_KEY)

# Load YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    logger.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLOv5 model: {e}")
    exit(1)

# Define the target animal and bird classes
TARGET_CLASSES = ['cat', 'dog', 'cow', 'horse', 'sheep', 'bird', 'deer', 'elephant', 'zebra', 'giraffe', 'lion', 'tiger']

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def send_to_gemini(image_path, detection_info):
    try:
        image = Image.open(image_path)
        gemini_model = genai.GenerativeModel(model_name='gemini-pro-vision')

        prompt = f"This image is captured in an airfield. Here are the details about the detected creatures:\n{detection_info}\nProvide full information in points, including count, type, height, and location as per airfield layout. Also, include information on probable hazardous birds with risk index and strike history."

        response = gemini_model.generate_content([prompt, image])
        name = gemini_model.generate_content(['only provide the name of animal in image without any additional information just like one word answer without any symols', image])
        return (to_markdown(response.text), name.text)
    except Exception as e:
        logger.error(f"Failed to send data to Gemini model: {e}")
        return None

def classify_frame(frame):
    try:
        results = model(frame)
        return results
    except Exception as e:
        logger.error(f"Failed to classify frame: {e}")
        return None

def extract_info(results):
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    detection_info = []
    for label, coord in zip(labels, coords):
        class_name = model.names[int(label)]
        if class_name in TARGET_CLASSES:
            x1, y1, x2, y2, conf = coord
            detection_info.append(f"{class_name}: {conf:.2f}, at [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
    return "\n".join(detection_info) if detection_info else None

# Create a folder to store the detected animal images
output_folder = 'data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open a connection to the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logger.error("Error: Could not open video stream.")
    exit(1)

# Main loop to capture frames and process them
frame_count = 0
while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Could not read frame from video stream.")
            break

        # Classify the frame
        results = classify_frame(frame)
        if results is None:
            continue

        detection_info = extract_info(results)
        if detection_info:
            # Save the detected animal image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(output_folder, f"animal_{frame_count}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            logger.info(f"Saved detected animal image to {image_path}")

            # Send the image and detection info to Gemini LLM
            response, name = send_to_gemini(image_path, detection_info)
            if response:
                logger.info(f"Frame {frame_count} sent: \n{response}")

            frame_count += 1
            # Automatically stop processing after detecting an animal
            logger.info("Animal detected and processed. Stopping...")
            
            tts=gTTS(text=f'please pay attention {name} spotted at airfield, full details are live', lang="en")
            filename = 'animal.mp3'
            tts.save(filename)
            playsound(os.path.abspath(filename))

            os.remove(os.path.abspath(filename))
            # winsound.PlaySound("sound.wav",winsound.SND_ASYNC)

        # Display the resulting frame with detections (optional)
        annotated_frame = np.squeeze(results.render())
        cv2.imshow('Frame', annotated_frame)

        # Press 'q' to exit (optional, kept for manual override)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        logger.error(f"Error during processing: {e}")

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
logger.info("Video capture released and windows closed.")
