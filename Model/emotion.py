#Import Dependencies
import os
import re
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from itertools import cycle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model,Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve,auc, confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from tensorflow.keras.layers import Dropout,Flatten,BatchNormalization,Dense,MaxPooling2D,Conv2D,Input,Activation,Add

#Define dataset path
dataset_folder = 'New_Data/'
sub_folders = os.listdir(dataset_folder)

#Declare folder names as labels and images within
i = 0
last = []
images = []
labels = []
tmp = sub_folders

#Read and set index 0:Happy and 1:Sad folders in dataset folder
for sub_folder in sub_folders:
  sub_folder_idx = tmp.index(sub_folder)
  label = sub_folder_idx
  
  path = dataset_folder+'/'+sub_folder
  sub_folder_images = os.listdir(path)
  
  #Read images in sub folder, one by one
  for image in sub_folder_images:
    image_path = path+'/'+image
    #print(image_path+"\t"+str(label))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    images.append(image)
    labels.append(label)
    i+=1
  last.append(i)
  
#Define x and y
images_x = np.array(images)
labels_y = np.array(labels)

#Encoding labels
labels_y_encoded = tf.keras.utils.to_categorical(labels_y, num_classes = 4)

#Split into 75 train and 25 test
X_train, X_test, Y_train, Y_test = train_test_split(images_x, labels_y_encoded, test_size = 0.30, random_state = 10)

#CNN Model Architecture
input_layer = Input(shape = (224, 224, 1))
conv1 = Conv2D(32, (3, 3), padding = 'same', strides = (1, 1), kernel_regularizer = l2(0.001))(input_layer)
conv1 = Dropout(0.1)(conv1)
conv1 = Activation('relu')(conv1)
pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), padding = 'same', strides = (1, 1), kernel_regularizer = l2(0.001))(pool1)
conv2 = Dropout(0.1)(conv2)
conv2 = Activation('relu')(conv2)
pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), padding = 'same', strides = (1, 1), kernel_regularizer = l2(0.001))(pool2)
conv3 = Dropout(0.1)(conv3)
conv3 = Activation('relu')(conv3)
pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), padding = 'same', strides = (1, 1), kernel_regularizer = l2(0.001))(pool3)
conv4 = Dropout(0.1)(conv4)
conv4 = Activation('relu')(conv4)
pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

flatten = Flatten()(pool4)
dense_1 = Dense(128, activation = 'relu')(flatten)
drop_1 = Dropout(0.2)(dense_1)
#Change to softmax for 4 classes
output = Dense(4, activation = 'softmax')(drop_1) 

#Compile Model
model = Model(inputs = input_layer, outputs = output)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
#model.summary()

#Configure Checkpoint Model
fle_s = 'Model/Output/Emotion_Gender_Model.keras'
checkpointer = ModelCheckpoint(fle_s, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', save_freq = 'epoch')
callback_list = [checkpointer]

save = model.fit(X_train, Y_train, batch_size = 32, validation_data = (X_test, Y_test), epochs = 100, callbacks = [callback_list])

#Checking train and test loss and accuracy values from above neural network
train_loss = save.history['loss']
test_loss = save.history['val_loss']
train_accuracy = save.history['accuracy']
test_accuracy = save.history['val_accuracy']

#Plotting line chart to visualize loss and accuracy values by epochs
fig, ax = plt.subplots(ncols = 2, figsize = (15, 7))
ax = ax.ravel()
ax[0].plot(train_loss, label = 'Train Loss', color = 'royalblue', marker = 'o', markersize = 5)
ax[0].plot(test_loss, label = 'Test Loss', color = 'orangered', marker = 'o', markersize = 5)
ax[0].set_xlabel('Epochs', fontsize = 14)
ax[0].set_ylabel('Categorical Crossentropy', fontsize = 14)
ax[0].legend(fontsize = 14)
ax[0].tick_params(axis = 'both', labelsize = 12)

ax[1].plot(train_accuracy, label = 'Train Accuracy', color = 'royalblue', marker = 'o', markersize = 5)
ax[1].plot(test_accuracy, label = 'Test Accuracy', color = 'orangered', marker = 'o', markersize = 5)
ax[1].set_xlabel('Epochs', fontsize = 14)
ax[1].set_ylabel('Accuracy', fontsize = 14)
ax[1].legend(fontsize = 14)
ax[1].tick_params(axis = 'both', labelsize = 12)
fig.suptitle(x = 0.5, y = 0.92, t = "Lineplots Showing Loss and Accuracy of CNN Model by Epochs", fontsize = 16)
plt.show()

# #Plotting confusion matix and roc curve of model
# dir = "Model/Output/Emotion_Model.keras"
# model = load_model(dir)
# best_model = model

# Y_pred = best_model.predict(X_test)
# Y_test_labels = np.argmax(Y_test, axis = 1)
# Y_pred_labels = np.argmax(Y_pred, axis = 1)

# cm = confusion_matrix(np.argmax(Y_test, axis = 1), np.argmax(Y_pred, axis = 1))
# disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Happy_Female", "Sad_Male", "Happy_Male", "Sad_Female",])
# disp.plot()
# plt.title('My CNN Emotion Classifer')
# plt.ylabel('Actual class')
# plt.xlabel('Predicted class')

# plt.gca().xaxis.set_label_position('top')
# plt.gca().xaxis.tick_top()

# plt.show()

# #ROC Curve
# new_label = ['Happy_Female', 'Sad_Male', 'Happy_Male', 'Sad_Female']
# final_label = new_label
# new_class = 4

# #Ravel flatten array into one vector
# y_pred_ravel = Y_pred.ravel()
# lw = 2

# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(new_class):
#     fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
    
# colors = cycle(['red', 'black', 'yellow', 'purple'])
# for i, color in zip(range(new_class), colors):
#     plt.plot(fpr[i], tpr[i], color = color, lw = lw,
#              label = 'ROC curve of class {0}'''.format(final_label[i]))
    
# plt.plot([0, 1], [0, 1], 'k--', lw = lw)
# plt.xlim([0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc = "lower right")
# plt.show()