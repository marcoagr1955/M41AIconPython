import cv2

import mediapipe as mp


# Inicializar Mediapipe FaceMesh

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Inicializar utilidades para dibujar

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# Inicializar OpenCV para captura de video

cap = cv2.VideoCapture(0)


while cap.isOpened():

    success, frame = cap.read()

    if not success:

        print("No se puede acceder a la cámara")

        break


    # Convertir la imagen a RGB

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Deshabilitar la escritura de la imagen para mejorar el rendimiento

    image.flags.writeable = False


    # Procesar la imagen y encontrar las mallas faciales

    results = face_mesh.process(image)


    # Volver a habilitar la escritura de la imagen

    image.flags.writeable = True


    # Convertir la imagen a BGR (para OpenCV)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # Dibujar las mallas faciales

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(

                image=image,

                landmark_list=face_landmarks,

                connections=mp_face_mesh.FACEMESH_TESSELATION,

                landmark_drawing_spec=drawing_spec,

                connection_drawing_spec=drawing_spec)


    # Mostrar el resultado en la ventana de video

    cv2.imshow('Reconocimiento de Gestos Faciales', image)


    # Salir si se presiona la tecla 'q'

    if cv2.waitKey(5) & 0xFF == ord('q'):

        break


# Liberar los recursos

cap.release()

cv2.destroyAllWindows()