#Import Dependencies
import cv2
import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt

#Declare Mediapipe Face Mesh
draw_me = mp.solutions.drawing_utils
mesh_face_me = mp.solutions.face_mesh
draw_styles_me = mp.solutions.drawing_styles

spec_draw = draw_me.DrawingSpec(thickness=1, circle_radius=1)

#Creating Face Mesh
with mesh_face_me.FaceMesh(
    static_image_mode = True,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5) as mesh_face:
    
    picture = cv2.imread('Model/Photos/6happy.png')
    if picture is not None:
        
        outcomes = mesh_face.process(picture)

        if outcomes.multi_face_landmarks:
            
            #Taking Copy of Picture
            picture_annotated = picture.copy()
    
            #Drawing Face Landmarks
            draw_me.draw_landmarks(
                image = picture_annotated,
                landmark_list = outcomes.multi_face_landmarks[0],
                connections=mesh_face_me.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = draw_styles_me.get_default_face_mesh_tesselation_style())

            #Saving Picture
            cv2.imwrite('Model/Photos/6happy.png', picture_annotated)

#Reopening Picture
pic = Image.open('Model/Photos/6happy.png')
pic.show()

#Mesh Face
mesh_face_me = mp.solutions.face_mesh
mesh_face = mesh_face_me.FaceMesh()

#Face Landmarks
outcome = mesh_face.process(picture)

if outcome.multi_face_landmarks:
    h, w, _ = picture.shape
    
    for facial_landmarks in outcome.multi_face_landmarks:
        for i in range(0, 468):
            point1 = facial_landmarks.landmark[i]
            x = int(point1.x * w)
            y = int(point1.y * h)
            cv2.circle(picture, (x, y), 5, (100, 100, 0), -1)
    
    plt.imshow(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
    plt.show()