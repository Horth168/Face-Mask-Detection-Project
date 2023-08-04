from imutils import paths
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy
import os

#Import the dataset and the image path
dataset=r'C:\Users\MSI-PC\Desktop\Face Mask Detection Project\dataset'

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths=list(paths.list_images(dataset))
data=[]
labels=[]

# loop over the image paths
for i in imagePaths:
    # extract the class label from the filename
    label=i.split(os.path.sep)[-2]
    # load the input image (224x224) and preprocess it
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data and labels to NumPy arrays
data=np.array(data,dtype='float32')
labels=np.array(labels)

# perform one-hot encoding on the labels
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

# partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing
train_X,test_X,train_Y,test_Y=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=10)
# construct the training image generator for data augmentation
aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')
# load the MobileNetV2 network, ensuring the head FC layer sets are
baseModel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

print(baseModel.summary())

# construct the head of the model that will be placed on top of the base model
headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

# place the head FC model on top of the base model (this will become the actual model we will train)
model=Model(inputs=baseModel.input,outputs=headModel)

# loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable=False

print(model.summary())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
learning_rate=0.001
Epochs=35
BS=21

# compile our model
print("[INFO] compiling model...")
opt=tf.keras.optimizers.legacy.Adam(learning_rate,decay=learning_rate/Epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

# train the head of the network
print("[INFO] training head...")
H=model.fit(
    aug.flow(train_X,train_Y,batch_size=BS),
    steps_per_epoch=len(train_X)//BS,
    validation_data=(test_X,test_Y),
    validation_steps=len(test_X)//BS,
    epochs=Epochs
)

# make predictions on the testing set
print("[INFO] evaluating network...")
predict=model.predict(test_X,batch_size=BS)
# for each image in the testing set we need to find the index of the-
# label with corresponding largest predicted probability
predict=np.argmax(predict,axis=1)
# show a nicely formatted classification report
print(classification_report(test_Y.argmax(axis=1),predict,target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
#model.save(r'C:\Users\MSI-PC\Desktop\Face Mask Detection Project\mask_detector.model')
model.save('mobilenet_v2.h5')

# plot the training loss and accuracy
N = Epochs
plt.style.use("ggplot")
plt.figure(1)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(r'C:\Users\MSI-PC\Desktop\Face Mask Detection Project\plot_loss.png')
plt.figure(2)
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(r'C:\Users\MSI-PC\Desktop\Face Mask Detection Project\plot_accuracy.png')
plt.show()

