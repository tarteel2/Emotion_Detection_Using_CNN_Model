#Import Dependencies
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import Image
from keras.models import load_model
from keras.preprocessing import image

model = load_model("Model/Output/Emotion_Model.keras")

#Creating Two Class Dictionary
emotion_classes = {0:'Happy', 1:'Sad'}

#Define Mediapipe Face Mesh
draw_me = mp.solutions.drawing_utils
mesh_face_me = mp.solutions.face_mesh
draw_styles_me = mp.solutions.drawing_styles

spec_draw = draw_me.DrawingSpec(thickness = 1, circle_radius = 1)

with mesh_face_me.FaceMesh(
    static_image_mode = True,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5) as mesh_face:

    #Read Downloaded Test Image In OpenCV
    test_img = cv2.imread('Model/Photos/3happy.png')
    
    #Convert Image To Gray Scale OpenCV
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    #Define Haar Cascade Classifier For Face Detection
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  
    #Detect face Using Haar Cascade Classifier
    faces_coordinates = face_classifier.detectMultiScale(gray_img, 1.3, 5)
    
    i = 0
    #Draw Rectangle Around Faces
    for (x, y, w, h) in faces_coordinates:
        i = i + 1
        #Draw Rectangle Around Face
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        #Crop Face From Image
        cropped_face = gray_img[y:y+h, x:x+w]
        final_image1 = cv2.resize(cropped_face, (224, 224), interpolation = cv2.INTER_AREA)
        final_image_array = np.array(final_image1)
        
        #Need 4th Dimension
        input_test = np.expand_dims(final_image_array, axis = 0) 
        output_test = emotion_classes[np.argmax(model.predict(input_test))]
        output_str = str(i) + ": " + output_test
        print(output_str)
        
        #Define OpenCV Font Style
        font = cv2.FONT_HERSHEY_SIMPLEX
        col = (0, 0, 255)
        cv2.putText(test_img, output_test, (x, y), font, 2, col, 2)
    
        outcomes = mesh_face.process(test_img)

        if outcomes.multi_face_landmarks:
            for face_landmarks in outcomes.multi_face_landmarks:
                #Drawing Face Landmarks
                draw_me.draw_landmarks(
                    image = test_img,
                    landmark_list = face_landmarks,
                    connections = mesh_face_me.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = draw_styles_me.get_default_face_mesh_tesselation_style())

    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.show()