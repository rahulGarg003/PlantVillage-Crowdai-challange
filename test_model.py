from tensorflow.keras.models import load_model
import cv2
import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import time


img_width, img_height = 128, 128
train_data_dir = "./crowdai"
validation_data_dir = "./crowdai"
nb_train_samples = 5
nb_validation_samples = 5
batch_size = 4
epochs = 10



model = load_model("Vgg_transfer.h5")

model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])


# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

board = TensorBoard(log_dir="./log_vgg{}/".format(time.time()),
                    write_graph=True,
                    batch_size=batch_size,
                    write_images=True,)


# Train the model
print(model.evaluate_generator(validation_generator,steps=1))