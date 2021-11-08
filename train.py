import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
import glob
import os
import cv2
import random
from sklearn.model_selection import train_test_split

data=[]

labels=[]

image_files = [f for f in glob.glob(r'C:\Users\Dragan\Desktop\gender_detection\gender_detection\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

i=9
for img in image_files:

    image = cv2.imread(img)
   
    image = cv2.resize(image, (96,96))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2] # C:\Files\gender_dataset_face\woman\face_1162.jpg
  
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append([label])
    
data = np.array(data, dtype="float") / 255.0
labels= np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


model= Sequential([
    
    Conv2D(filters=32,kernel_size=(3,3), activation='relu', padding='same', input_shape=(96,96,3)),
    MaxPooling2D(pool_size=(2,2), strides=2),
    Dropout(0.25),
    
    
    Conv2D(filters=64,kernel_size=(3,3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=2),
    Dropout(0.25),
    
    
    Conv2D(filters=64,kernel_size=(3,3), activation='relu', padding='same',kernel_regularizer='l2'),
    MaxPooling2D(pool_size=(2,2), strides=2),
    Dropout(0.25),
    
    Flatten(),
    Dense(units=2,activation='softmax')  
    
    ])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['acc'])

H = model.fit_generator(aug.flow(trainX, trainY, 50),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // 50,
                        epochs=500, verbose=1)

model.save('gender_detection.model')


plt.style.use("ggplot")
plt.figure()
N = 500
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plotgender.png')

