from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix

from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd
import itertools
import keras

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.68-0.73.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion = ["angry", "happy", "sad", "surprised","neutral"]
img_rows, img_cols = 48, 48
num_classes = 5

data = pd.read_csv('fer2013/fer2013.csv', delimiter=',')
data_test = data[27215:]
# data_test = data[32300:]
y_test = data_test['emotion'].values
print('Len ', len(y_test))
x_test = np.zeros((y_test.shape[0], 48*48))
for i in range(y_test.shape[0]):
    try:
        x_test[i] = np.fromstring(data_test['pixels'][27215+i], dtype=int, sep=' ')
        # x_test[i] = np.fromstring(data_test['pixels'][32300+i], dtype=int, sep=' ')
    except:
        pass

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Loading Models...')

model = []
model.append(emotion_classifier)

print("Loading Complete!")

def plot_confusion_matrix(cm):
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Private Test Data\n Haar + Mini-Xception')
    plt.colorbar()
    tick_marks = np.arange(len(emotion))
    plt.xticks(tick_marks, emotion)
    plt.yticks(tick_marks, emotion)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Emotions')
    plt.xlabel('Predicted Emotions')
    plt.tight_layout()

p = np.zeros((y_test.shape[0],5))
# print('y_test ',y_test.shape)
# p = np.zeros((1,14))
# p[:,0:5] = model[0].predict()
# print('p ',p.shape)
y_pred = model[0].predict(x_test)
yp = np.argmax(y_pred, axis=1)
yt = np.argmax(y_test, axis=1)
cm = confusion_matrix(yt, yp)
plot_confusion_matrix(cm)
plt.savefig("fer2013/cm.png")
plt.show()