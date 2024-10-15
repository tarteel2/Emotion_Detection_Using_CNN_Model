#Import Dependencies
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from keras.models import load_model

model = load_model("Model/Output/Emotion_Model.keras")

emotion_classes = {0:'Happy', 1:'Sad'}

#Declare Mediapipe Face Mesh
draw_me = mp.solutions.drawing_utils
mesh_face_me = mp.solutions.face_mesh
draw_styles_me = mp.solutions.drawing_styles

#Load Face Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Start Capturing Video
cap = cv2.VideoCapture(0)

with mesh_face_me.FaceMesh(
    static_image_mode = True,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5) as mesh_face:
    
    while True:
        #Capture Frame-By-Frame
        ret, frame = cap.read()

        #Convert Frame To Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Detect Faces In Frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            #Extract Face ROI (Region of Interest)
            face_roi = gray_frame[y:y + h, x:x + w]
            #Resize Image To Match Model's Expected Sizing
            img = cv2.resize(face_roi ,(224, 224))     
            img = img.reshape(-1, 224, 224, 1) 
            
            #Perform Emotion Analysis On Face ROI
            #Determine Emotion
            predicted_emotion = model.predict(img)
            index = int(np.argmax(predicted_emotion))

            #Draw Rectangle Around Face & Label With Predicted Emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_classes[index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            #Change BGR frame to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            outcomes = mesh_face.process(rgb_frame)
            
            if outcomes.multi_face_landmarks:
                for face_landmarks in outcomes.multi_face_landmarks:
                    #Drawing Face Landmarks
                    draw_me.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmarks,
                        connections = mesh_face_me.FACEMESH_TESSELATION,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = draw_styles_me.get_default_face_mesh_tesselation_style())
                    
        #Display Resulting Frame
        cv2.imshow('Real-Time Emotion And Face Mesh Detection', frame)

        #Press 'q' To Exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

#Release Capture & Close All Windows
cap.release()
cv2.destroyAllWindows()