#!/usr/bin/env python
__author__ = 'ziyan.yin'

import cv2
import imageutils


def _grey(image):
    img = imageutils.load_image(image, method='CV2')
    if img.ndim == 3:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey = img
    return grey


def detect_faces(image):
    grey = _grey(image)
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
    faces = face_cascade.detectMultiScale(grey, 1.1, 3)
    result = []
    for (x, y, w, h) in faces:
        result.append((x, y, x+w, y+h))
    return result


def _detect_organ_on_face(image, cascade, scale, neighbor, flag=4, min_size=0.1, max_size=1.0):
    faces = detect_faces(image)
    grey = _grey(image)
    eye_cascade = cv2.CascadeClassifier(cascade)
    result = []
    for (x1, y1, x2, y2) in faces:
        face = grey[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(
            face, scale, neighbor, flag,
            (int(min_size * (x2 - x1)), int(min_size * (x2 - x1))) if min_size else None,
            (int(max_size * (x2 - x1)), int(max_size * (x2 - x1))) if max_size else None,
        )
        for (x, y, w, h) in eyes:
            result.append((x1 + x, y1 + y, x1 + x + w, y1 + y + h))
    return result


def detect_eyes(image):
    return _detect_organ_on_face(image, "data/haarcascade_eye.xml", 1.1, 3, 4, 0.16, 0.32)


def detect_left_eye(image):
    eyes = detect_eyes(image)
    times = 0
    temp = (0, 0, 0, 0)
    result = []
    for eye in eyes:
        if times == 0:
            temp = eye
        elif times == 1:
            if temp[0] > eye[0]:
                result.append(eye)
            else:
                result.append(temp)
        times += 1
        if times > 1:
            times = 0
    return result


def detect_right_eye(image):
    eyes = detect_eyes(image)
    times = 0
    temp = (0, 0, 0, 0)
    result = []
    for eye in eyes:
        if times == 0:
            temp = eye
        elif times == 1:
            if temp[0] < eye[0]:
                result.append(eye)
            else:
                result.append(temp)
        times += 1
        if times > 1:
            times = 0
    return result


def detect_mouth(image):
    return _detect_organ_on_face(image, "data/haarcascade_mcs_mouth.xml", 1.1, 3, 4, 0.3, 0.5)


if __name__ == '__main__':
    image = "d:/face.jpg"
    result = detect_eyes(image)
    img = imageutils.load_image(image, method='CV2')
    for (x1, y1, x2, y2) in result:
        mouth = img[y1:y2, x1:x2]
    cv2.imshow("eyes", img)
    cv2.waitKey()
