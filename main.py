import cv2
import numpy as np
from deepface import DeepFace
import time

fps_start_time = 0
fps = 0


face_detector = cv2.CascadeClassifier(
       cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Find haar cascade to draw bounding box around face
    _, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / (time_diff)
    fps_start_time = fps_end_time
    fps_text = "FPS: {:.2f}".format(fps)
    font = cv2.FONT_HERSHEY_PLAIN

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        # roi_gray_frame = gray_frame[y:y + h, x:x + w] # isolates the face region
        # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        try:
            analyze = DeepFace.analyze(frame, actions = ['emotion'])
            nigga = analyze[0]['dominant_emotion']
            print(nigga)
        except:
            print("no face")

    cv2.putText(frame, analyze[0]['dominant_emotion'], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(frame, fps_text, (10, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, nigga, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                 cv2.LINE_AA)

    cv2.imshow('video', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()