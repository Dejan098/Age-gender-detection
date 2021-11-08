import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense,SpatialDropout2D, GlobalAveragePooling2D
import glob
import os
import cv2
import random
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt


data=[]

labels=[]

image_files = [f for f in glob.glob(r'C:\Users\Dragan\Desktop\gender_detection\gender_detection\age_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)
i=0
for img in image_files:
    image = cv2.imread(img)
   
    image = cv2.resize(image, (96,96))
    image = img_to_array(image)
    data.append(image)
    
    labela = img.split(os.path.sep)[-2]
    
   # findAge=re.split(r'\_',labela)
   # yearOfBirth  = re.split(r'\-', findAge[1])[0]
   # yearPhotoTaken = re.split(r'\.', findAge[2])[0]
   #age = int(yearPhotoTaken) - int(yearOfBirth)
    
    #labelica = img.split(os.path.sep)[-2]
    labela=int(labela)
   
    if (labela>0 and labela<=10):
         label=0
    if (labela>10 and labela<=20):
         label=2
    if (labela>20 and labela<=30):
         label=2    
    if (labela>30 and labela<=40):
         label=3 
    if (labela>40 and labela<=50):
         label=4 
    if (labela>50 and labela<=60):
         label=5 
    if (labela>60 and labela<=70):
         label=6
    if (labela>70 and labela<=80):
         label=7
    

          

    
    
        

    
        
    labels.append([label])


data = np.array(data, dtype="float") / 255.0
labels= np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=8)
print(trainY)
testY = to_categorical(testY, num_classes=8)


aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")



age_model= Sequential([
    
    
Conv2D(16, (3, 3), padding='same', input_shape=(96,96,3), activation='relu'),
BatchNormalization(axis=1),
MaxPooling2D(pool_size=(2,2)),
Dropout(0.2),

Conv2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer='l2'),
BatchNormalization(axis=1),
MaxPooling2D(pool_size=(2,2)),
Dropout(0.2),


Conv2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer='l2'),
BatchNormalization(axis=1),
MaxPooling2D(pool_size=(2,2)),
Dropout(0.3),

Flatten(),

#S3 Fully connected 

Dense(units=8, activation='softmax')

])


age_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

H = age_model.fit_generator(aug.flow(trainX, trainY, 10),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // 10,
                        epochs=170, verbose=1)

age_model.save('age_detection.model')

plt.style.use("ggplot")
plt.figure()
N = 170 
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('age170epochs.png')