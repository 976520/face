import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                            (landmarks.part(37).x, landmarks.part(37).y),
                            (landmarks.part(38).x, landmarks.part(38).y),
                            (landmarks.part(39).x, landmarks.part(39).y),
                            (landmarks.part(40).x, landmarks.part(40).y),
                            (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        
        right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                             (landmarks.part(43).x, landmarks.part(43).y),
                             (landmarks.part(44).x, landmarks.part(44).y),
                             (landmarks.part(45).x, landmarks.part(45).y),
                             (landmarks.part(46).x, landmarks.part(46).y),
                             (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
    
    cv2.imshow('eye', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
