# Function to detect faces in a grascale image returns its top right corner and resized image of 48x48

import numpy as np
import cv2
import matplotlib.pyplot as plt
# for haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# for dnn based face detection
modelFile = "saved_model/opencv_face_detector_uint8.pb"
configFile = "saved_model/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def dnn_faces(img, thres):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    predict = net.forward()

    faces_detected = []
    [h, w] = img.shape[:2]
    for i in range(predict.shape[2]):
        confidence = predict[0, 0, i, 2]
        if confidence > thres:
            x1 = predict[0, 0, i, 3] * w
            y1 = predict[0, 0, i, 4] * h
            x2 = predict[0, 0, i, 5] * w
            y2 = predict[0, 0, i, 6] * h
            wf = x2 - x1
            hf = y2 - y1
            faces_detected.append((int(x1), int(y1), int(wf), int(hf)))

    return faces_detected

def get_faces(img, method='haar'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = []

    if (method == 'dnn'):
        faces_detected = dnn_faces(img, 0.5)
    elif (method == 'haar'):
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for i, (x,y,w,h) in enumerate(faces_detected):
        my = int(y + h/2)
        mx = int(x + w/2)

        if h<w:
            c = int(h/2)
        else:
            c = int(w/2)

        face = gray[my-c:my+c, mx-c:mx+c]
        try:
            face_48 = cv2.resize(face,(48, 48), interpolation = cv2.INTER_CUBIC)
            faces.append((face_48,x,y,w,h))
            # faces.append(())
        except:
            pass

    return faces

# img = cv2.imread('img/test.jpeg', cv2.IMREAD_COLOR)
# cv2.imshow('image',img)
# faces=get_faces(img,method='haar')
# print(len(faces))
# print(faces)
# for i, (face,x,y,w,h) in enumerate(faces):
#     print(i,face)
# tl=(x,y)
# br=(x+w,y+h)
# img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
# img = cv2.putText(img, 'label', tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),2)
# plt.imshow(img)
# plt.show()
# while True:
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()