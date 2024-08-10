import cv2

import mediapipe as mp

import numpy as np


# Inicializar Mediapipe FaceMesh

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Inicializar utilidades para dibujar

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# Inicializar OpenCV para captura de video

cap = cv2.VideoCapture(0)



def calcular_distancia(punto1, punto2):

    return np.linalg.norm(np.array(punto1) - np.array(punto2))



while cap.isOpened():

    success, frame = cap.read()

    if not success:

        print("No se puede acceder a la cámara")

        break


    # Convertir la imagen a RGB

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False


    # Procesar la imagen y encontrar las mallas faciales

    results = face_mesh.process(image)


    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(

                image=image,

                landmark_list=face_landmarks,

                connections=mp_face_mesh.FACEMESH_TESSELATION,

                landmark_drawing_spec=drawing_spec,

                connection_drawing_spec=drawing_spec)


            # Coordenadas para los puntos clave de la boca

            labio_superior = face_landmarks.landmark[13]  # Punto del labio superior

            labio_inferior = face_landmarks.landmark[14]  # Punto del labio inferior


            # Convertir a coordenadas absolutas

            altura, ancho, _ = image.shape

            labio_superior = (int(labio_superior.x * ancho), int(labio_superior.y * altura))

            labio_inferior = (int(labio_inferior.x * ancho), int(labio_inferior.y * altura))


            # Calcular la distancia entre los labios

            distancia_boca = calcular_distancia(labio_superior, labio_inferior)


            # Umbral para determinar si la boca está abierta

            umbral_boca_abierta = 20  # Ajustar según sea necesario


            if distancia_boca > umbral_boca_abierta:

                cv2.putText(image, 'Boca Abierta!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print("La boca esta abierta")


    # Mostrar el resultado en la ventana de video

    cv2.imshow('Reconocimiento de Gestos Faciales', image)


    # Salir si se presiona la tecla 'q'

    if cv2.waitKey(5) & 0xFF == ord('q'):

        break


# Liberar los recursos

cap.release()

cv2.destroyAllWindows()