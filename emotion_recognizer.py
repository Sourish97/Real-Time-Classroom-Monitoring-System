from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from detectfaces import get_faces
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import matplotlib.pyplot as plt
import subprocess
from keras import backend as K
import importlib



import tensorflow as tf
from scipy import misc
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import pandas as pd

df = pd.read_csv('Custom/Evaluation.csv')
#
img_rows, img_cols = 48, 48

if K.backend() != 'theano':
    os.environ['KERAS_BACKEND'] = 'theano'
    importlib.reload(K)

emotion = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (255,0,0)
box_color = (255,245,152)



Exprmodel = []
print('Loading Models...')
print('0/3')
for i in range(2):
    m = load_model('saved_model/' + 'cnn'+str(i)+'.h5')
    print(str(i+1) + '/3')
    Exprmodel.append(m)

m = load_model('saved_model/ensemble.h5')
Exprmodel.append(m)
print('3/3')

print("Loading Complete!")

def predict(x):
    x_rev = np.flip(x, 1)
    x = x.astype('float32')
    x_rev = x_rev.astype('float32')
    x /= 255
    x_rev /= 255
    p = np.zeros((1, 10))
    p[:,0:5] = Exprmodel[0].predict(x.reshape(1,48,48,1))
    p[:,5:10] = Exprmodel[1].predict(x_rev.reshape(1,48,48,1))
    pre = Exprmodel[2].predict(p)
    return pre
#
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'det')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        

        HumanNames = [df['Name'][i] for i in range(len(df['Name']))]    #train human name
        print()

        t_students = {HumanNames[i]:{'focus':0, 'distract':0} for i in range(len(HumanNames))}
        

        print('Loading feature extraction model')
        modeldir = '20180402-114759/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = 'my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            video_capture.open()
        c = 0

        print('Start Recognition!')
        prevTime = 0
        start_class = time.time()
        while True:
            ret, frame = video_capture.read()
            img = frame
            t_focused=0 
            t_distracted = 0 

            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    faces = get_faces(img, method='haar')

                    for i,(face,x,y,w,h) in enumerate(faces):
                        pre = predict(face)
                        k = np.argmax(pre)
                        st = time.time()
                        start = time.time()
                        tl=(x,y)
                        coords = (x,y-2)
                        br=(x+w,y+h)
                        # frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                        # txt = name+' '+emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%]'
                        # cv2.putText(frame, txt, coords, font, 0.5,(0,255,0),1,cv2.LINE_AA)
                        emb_array = np.zeros((1, embedding_size))

                        try:
                            bb[i][0] = x #det[i][0]
                            bb[i][1] = y #det[i][1]
                            bb[i][2] = x+w #det[i][2]
                            bb[i][3] = y+h #det[i][3]
                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[0] = facenet.flip(cropped[0], False)
                            scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                            scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled[0] = facenet.prewhiten(scaled[0])
                            scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), box_color, 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                        except:
                            pass
                        # print('result: ', best_class_indices[0])
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                en = time.time()
                                d= en - st
                                if emotion[k] in ['Angry', 'Happy', 'Sad', 'Surprise']:
                                    box_color = (0,0,255)
                                    txt = result_names+' '+emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%] | Distracted'
                                    t_students[result_names]['distract'] = t_students[result_names]['distract'] + d

                                else:
                                    box_color = (255,245,152)
                                    txt = result_names+' '+emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%] | Focused'
                                    t_students[result_names]['focus'] = t_students[result_names]['focus'] + d
                                #txt = result_names+' '+emotion[k] + ' [' + str(int(pre[0,k]*100)) + '%]'
                                cv2.putText(frame, txt, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, text_color, thickness=2, lineType=2)
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            print('sec',sec)
            # fps = 1 / (sec)
            # str1 = 'FPS: %2.3f' % fps
            # text_fps_x = len(frame[0]) - 150
            # text_fps_y = 20
            # cv2.putText(frame, str1, (text_fps_x, text_fps_y),
            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
        
            # # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                end_class = time.time()
                t_total = end_class - start_class
                print(t_students)
                #df['t_total'][0] = df['t_total'][0] + t_total
                #df.to_csv('Custom/Evaluation.csv',index = False)
                break

        video_capture.release()
        cv2.destroyAllWindows()   