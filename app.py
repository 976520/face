import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import dlib

def create_lip_reading_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # softmax
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def detect_and_process_lips():
    cap = cv2.VideoCapture(0)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    model = create_lip_reading_model()
    # model.load_weights('lip_reading_model.h5')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            
            # 48-68
            lip_points = []
            for n in range(48, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                lip_points.append((x, y))
            
            lip_points = np.array(lip_points)
            x, y, w, h = cv2.boundingRect(lip_points)
            lip_roi = gray[y:y+h, x:x+w]
            
            if lip_roi.size > 0:
                lip_roi = cv2.resize(lip_roi, (100, 100))
                lip_roi = lip_roi.reshape(1, 100, 100, 1)
                
                prediction = model.predict(lip_roi)
                predicted_word = ["zero", "one", "two", "three", "four", 
                                "five", "six", "seven", "eight", "nine"][np.argmax(prediction)]
                
                cv2.putText(frame, predicted_word, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            cv2.polylines(frame, [lip_points], True, (0, 255, 0), 2)
        
        cv2.imshow('Lip Reading', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_process_lips()
