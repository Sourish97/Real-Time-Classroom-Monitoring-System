import numpy as np
import cv2
from detectfaces import get_faces
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import matplotlib.pyplot as plt
import face_recognition
import time
import pandas as pd
import datetime

now = datetime.datetime.now()
date = str(now.strftime("%d-%m-%Y %H:%M")).split(' ')[0].replace('-','/').encode()
# print(date)
# Load a sample picture and learn how to recognize it.
sourish_image = face_recognition.load_image_file("images/sourish.jpg")
sourish_face_encoding = face_recognition.face_encodings(sourish_image)[0]

# Load a second sample picture and learn how to recognize it.
gaurav_image = face_recognition.load_image_file("images/gaurav.jpg")
gaurav_face_encoding = face_recognition.face_encodings(gaurav_image)[0]

# Load a second sample picture and learn how to recognize it.
yash_image = face_recognition.load_image_file("images/yash.jpg")
yash_face_encoding = face_recognition.face_encodings(yash_image)[0]

danesh_image = face_recognition.load_image_file("images/danesh.jpg")
danesh_face_encoding = face_recognition.face_encodings(danesh_image)[0]

dhanesh_image = face_recognition.load_image_file("images/dhanesh.jpg")
dhanesh_face_encoding = face_recognition.face_encodings(dhanesh_image)[0]

girish_image = face_recognition.load_image_file("images/girish.jpg")
girish_face_encoding = face_recognition.face_encodings(girish_image)[0]

harsh_image = face_recognition.load_image_file("images/harsh.jpg")
harsh_face_encoding = face_recognition.face_encodings(harsh_image)[0]

harshad_image = face_recognition.load_image_file("images/harshad.jpg")
harshad_face_encoding = face_recognition.face_encodings(harshad_image)[0]

karishma_image = face_recognition.load_image_file("images/karishma.jpg")
karishma_face_encoding = face_recognition.face_encodings(karishma_image)[0]

neeral_image = face_recognition.load_image_file("images/neeral.jpg")
neeral_face_encoding = face_recognition.face_encodings(neeral_image)[0]

pavan_image = face_recognition.load_image_file("images/pavan.jpg")
pavan_face_encoding = face_recognition.face_encodings(pavan_image)[0]

pranav_image = face_recognition.load_image_file("images/pranav.jpg")
pranav_face_encoding = face_recognition.face_encodings(pranav_image)[0]

rahul_image = face_recognition.load_image_file("images/rahul.jpg")
rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]

raina_image = face_recognition.load_image_file("images/raina.jpg")
raina_face_encoding = face_recognition.face_encodings(raina_image)[0]

sanket_image = face_recognition.load_image_file("images/sanket.jpg")
sanket_face_encoding = face_recognition.face_encodings(sanket_image)[0]

saqlain_image = face_recognition.load_image_file("images/saqlain.jpg")
saqlain_face_encoding = face_recognition.face_encodings(saqlain_image)[0]

sakshi_mam_image = face_recognition.load_image_file("images/sakshi_mam.jpg")
sakshi_mam_encoding = face_recognition.face_encodings(sakshi_mam_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    sourish_face_encoding,
    gaurav_face_encoding,
    yash_face_encoding,
    danesh_face_encoding,
    dhanesh_face_encoding,
    girish_face_encoding,
    harsh_face_encoding,
    harshad_face_encoding,
    karishma_face_encoding,
    neeral_face_encoding,
    pavan_face_encoding,
    pranav_face_encoding,
    rahul_face_encoding,
    raina_face_encoding,
    sanket_face_encoding,
    saqlain_face_encoding,
    sakshi_mam_encoding
]
known_face_names = [
    "sourish",
    "gaurav",
    "yash",
    "danesh",
    "dhanesh",
    "girish",
    "harsh",
    "harshad",
    "karishma",
    "neeral",
    "pavan",
    "pranav",
    "rahul",
    "raina",
    "sanket",
    "saqlain",
    "sakshi_mam"
]

t_students = {known_face_names[i]:{'focus':0, 'distract':0, 'attendance':0} for i in range(len(known_face_names))}
df = pd.read_csv('Haar + Mini-Xception/Evaluation.csv')
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
attendance = []


img_rows, img_cols = 48, 48
text_color = (0,255,0)
box_color = (0,215,255)
emotion = ["angry" , "happy", "sad", "surprised", "neutral"]
font = cv2.FONT_HERSHEY_SIMPLEX
emotion_model_path = 'models/_mini_XCEPTION.68-0.73.hdf5'
print('Loading Models...')
model = load_model(emotion_model_path)

print("Loading Complete!")

def predict(x):
    x = x.astype('float32')
    x /= 255
    pre = model.predict(x.reshape(1,48,48,1))
    return pre

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

prevTime = 0
sum=0
sum1=0
start = time.time()

while(True):
    ret, img = cap.read()
    curTime = time.time()

    faces = get_faces(img, method='haar')
    for i,(face,x,y,w,h) in enumerate(faces):
        pre = predict(face)
        k = np.argmax(pre)
        # cv2.imshow('Image',img[y-20:y+h+20,x-20:x+w+20])
        # while True:
        #     if cv2.waitKey(20) & 0xFF == ord('q'):
        #         break
        name = ''
        try:
            small_frame = cv2.resize(img[y-20:y+h+20,x-20:x+w+20], (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # print('face_locations',face_locations)
                # print('face_encoding',face_encoding)
                    # See if the face is a match for the known face(s)

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])

                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    if(name!="Unknown"):
                        t_students[name]['attendance']=1

            # process_this_frame = not process_this_frame
        except:
            pass
        
        tl=(x,y)
        coords = (x,y-2)
        br=(x+w,y+h)
        try:
            if emotion[k] in ["angry" , "happy", "sad", "surprised"]:
                box_color = (0,0,255)
                txt = name+' '+emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%] | Distracted'
                if(name!="Unknown"):
                    t_students[name]['distract'] +=1
            else:
                box_color = (0,215,255)
                txt = name+' '+emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%] | Focused'
                if(name!="Unknown"):
                    t_students[name]['focus'] +=1
        except:
            pass
        # sum1+=1
        img = cv2.rectangle(img, tl, br, box_color, 2)
        cv2.putText(img, txt, coords, font, 0.8,text_color,1,cv2.LINE_AA)
        # cv2.imshow(str(i), face)
    # sec = curTime - prevTime
    # prevTime = curTime
    # print('sec',sec)
    # sum+=sec
    # fps=0
    # try:
    #     fps = 1 / (sec)
    # except:
    #     pass
    # str1 = 'FPS: %2.3f' % fps
    # text_fps_x = len(img[0]) - 150
    # text_fps_y = 20
    # cv2.putText(img, str1, (text_fps_x, text_fps_y),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
    
        # # c+=1
    cv2.imshow('Camera', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        end=time.time()
        total = end - start
        # total = total - 0.15*total
        for n in known_face_names:
            t_students[n]['focus']/=10
            t_students[n]['distract']/=10
            if(t_students[n]['attendance']==1):
                print(n,'Present')
                attendance.append('Present')
            else:
                print(n,'Absent')
                attendance.append('Absent')
            ind = df[df['Name'] == n].index.values.astype(int)
            df.loc[ind,'t_focused'] += t_students[n]['focus']
            df.loc[ind,'t_distracted'] += t_students[n]['distract']
        
        df.loc[0,'t_total']+=total
        print(t_students)
        print('total',total)

        for i in range(len(df['Name'])):
            status = df.loc[i,'t_focused']/df['t_total'][0]
            if(status >= 0.6):
                df.loc[i,'Status'] = 'Pass'
            else:
                df.loc[i,'Status'] = 'Fail'
        print(attendance)
        df[date] = attendance
        df.to_csv('Haar + Mini-Xception/Evaluation.csv',index = False)

        break

cap.release()
cv2.destroyAllWindows()