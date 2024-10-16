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
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model,Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve,auc, confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input,Add,Reshape

#Define Dataset Path
dataset_folder = 'New_Data/'
sub_folders = os.listdir(dataset_folder)

#Declare Folder Names as Labels and Images Within
i = 0
last = []
images = []
labels = []
tmp = sub_folders

#Read Sub Folders in Main Dataset Folder one at a time
for sub_folder in sub_folders:
  sub_folder_idx = tmp.index(sub_folder)
  label = sub_folder_idx
  
  path = dataset_folder+'/'+sub_folder
  sub_folder_images = os.listdir(path)
  
  #Read Images in Sub Folders, one by one
  for image in sub_folder_images:
    image_path = path+'/'+image
    #print(image_path+"\t"+str(label))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    images.append(image)
    labels.append(label)
    i+=1
  last.append(i)
  
#Define X and Y
images_x = np.array(images)
labels_y = np.array(labels)

#Encoding Labels
labels_y_encoded = tf.keras.utils.to_categorical(labels_y, num_classes = 2)

#Split Into 75 Train and 25 Test
X_train, X_test, Y_train, Y_test = train_test_split(images_x, labels_y_encoded, test_size = 0.25, random_state = 10)

#YOLO Model Architecture
input_layer = Input(shape = (224, 224, 1))
    
#Convolution Layers
x = layers.Conv2D(32, (3, 3), padding = 'same', activation = "relu", kernel_regularizer = l2(0.0001))(input_layer)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(64, (3, 3), padding = 'same', activation = "relu", kernel_regularizer = l2(0.0001))(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(128, (3, 3), padding = 'same', activation = "relu", kernel_regularizer = l2(0.0001))(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Conv2D(256, (3, 3), padding = 'same', activation = "relu", kernel_regularizer = l2(0.0001))(x)
x = layers.MaxPooling2D(pool_size = (2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation = "relu")(x)
x = layers.Dropout(0.5)(x)

#Detection Layer
output = Dense(2, activation = "softmax")(x)

#Compile YOLO Model
yolo_model = Model(inputs = input_layer, outputs = output)
yolo_model.compile(optimizer = Adam(learning_rate = 0.0001), loss = "categorical_crossentropy", metrics = ['accuracy'])
#yolo_model.summary()

#Configure Checkpoint Model
# fle_s = 'Model/Output/Emotion_Model.keras'
# checkpointer = ModelCheckpoint(fle_s, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', save_freq = 'epoch')
# callback_list = [checkpointer]

#save = yolo_model.fit(X_train, Y_train, batch_size = 32, validation_data = (X_test, Y_test), epochs = 100, callbacks = [callback_list])

# #Checking Train and Test Loss and Accuracy Values from Above Model
# train_loss = save.history['loss']
# test_loss = save.history['val_loss']
# train_accuracy = save.history['accuracy']
# test_accuracy = save.history['val_accuracy']

# #Plotting Line Chart to Visualize Loss and Accuracy Values by Epochs
# fig, ax = plt.subplots(ncols = 2, figsize = (15, 7))
# ax = ax.ravel()
# ax[0].plot(train_loss, label = 'Train Loss', color = 'royalblue', marker = 'o', markersize = 5)
# ax[0].plot(test_loss, label = 'Test Loss', color = 'orangered', marker = 'o', markersize = 5)
# ax[0].set_xlabel('Epochs', fontsize = 14)
# ax[0].set_ylabel('Categorical Crossentropy', fontsize = 14)
# ax[0].legend(fontsize = 14)
# ax[0].tick_params(axis = 'both', labelsize = 12)

# ax[1].plot(train_accuracy, label = 'Train Accuracy', color = 'royalblue', marker = 'o', markersize = 5)
# ax[1].plot(test_accuracy, label = 'Test Accuracy', color = 'orangered', marker = 'o', markersize = 5)
# ax[1].set_xlabel('Epochs', fontsize = 14)
# ax[1].set_ylabel('Accuracy', fontsize = 14)
# ax[1].legend(fontsize = 14)
# ax[1].tick_params(axis = 'both', labelsize = 12)
# fig.suptitle(x = 0.5, y = 0.92, t = "Lineplots Showing Loss and Accuracy of Yolo Model by Epochs", fontsize = 16)
# plt.show()

#Plotting Confusion Matix and ROC Curve of Model
dir = "Model/Output/Emotion_Model.keras"
model = load_model(dir)
best_model = model

Y_pred = best_model.predict(X_test)
Y_test_labels = np.argmax(Y_test, axis = 1)
Y_pred_labels = np.argmax(Y_pred, axis = 1)

cm = confusion_matrix(np.argmax(Y_test, axis = 1), np.argmax(Y_pred, axis = 1))
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Happy", "Sad"])
disp.plot()

plt.title('My Yolo Emotion Classifer')
plt.ylabel('Actual class')
plt.xlabel('Predicted class')

plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()

plt.show()

#ROC Curve
new_label = ['Happy', 'Sad']
final_label = new_label
new_class = 2

#Ravel flatten array into one vector
y_pred_ravel = Y_pred.ravel()
lw = 2

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(new_class):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
colors = cycle(['red', 'black'])
for i, color in zip(range(new_class), colors):
    plt.plot(fpr[i], tpr[i], color = color, lw = lw,
             label = 'ROC curve of class {0}'''.format(final_label[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw = lw)
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = "lower right")
plt.show()