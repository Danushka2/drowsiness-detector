# https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/

import cv2
import dlib
import numpy as np
from scipy.spatial import distance

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def calculate_eye_distance(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A + B) / (2.0 * C)
    return eye_aspect_ratio


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    success, frame = cap.read()

    original = frame.copy()
    cv2.putText(original, "Step 1", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (169, 169, 169), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayDuplicate = gray.copy()

    cv2.putText(gray, "Step 2", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (169, 169, 169), 2)
    cv2.putText(grayDuplicate, "Step 3", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (169, 169, 169), 2)

    faces = hog_face_detector(gray)

    for (i, rect) in enumerate(faces):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        cv2.rectangle(grayDuplicate, (x, y), (x + y, y + h), (255, 0, 0), 2)

    cv2.putText(frame, "Step 4", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (169, 169, 169), 2)

    for face in faces:
        face_landmarks = dlib_face_landmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_eye = calculate_eye_distance(leftEye)
        right_eye = calculate_eye_distance(rightEye)

        EYE = (left_eye + right_eye) / 2
        EYE = round(EYE, 2)
        if EYE < 0.26:
            cv2.putText(frame, "Drowsiness Alert!!", (20, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("Drowsy")
        print(EYE)

    imgStack = stackImages(0.6, ([original, gray], [grayDuplicate, frame]))
    cv2.imshow("Drowsiness Detection", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
