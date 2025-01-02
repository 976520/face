import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import dlib
from threading import Thread
import queue

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

frame_queue = queue.Queue(maxsize=2)
processed_frame_queue = queue.Queue(maxsize=2)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        left_eye = np.array(left_eye_coords, np.int32)
        right_eye = np.array(right_eye_coords, np.int32)
        
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
    
    return frame

def frame_processor():
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        if frame is None:
            break
        processed = process_frame(frame)
        processed_frame_queue.put(processed)

processor_thread = Thread(target=frame_processor)
processor_thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if not frame_queue.full():
        frame_queue.put(frame)
    
    if not processed_frame_queue.empty():
        processed_frame = processed_frame_queue.get()
        cv2.imshow('eye', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

frame_queue.put(None)
processor_thread.join()
cap.release()
cv2.destroyAllWindows()
