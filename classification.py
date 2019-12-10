
# Note: all the code below was executed in a jupyter notebook piece by piece 
# All the pieces are marked with a number (e.g. In[3] - means that the next section of the code was run after the first two)
# This code is used to train a CNN to recognise galaxy morphology (GalaxyZoo Challenge)
# Challenge can be accessed here: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

# Training information:
# Since the challenge is quite complex, the training had to be done on a powerful machine
# Microsoft Azure NV6 machine with NVIDIA Tesla K80 was used to perform this training
# It took around 25 minutes per epoch, and the complete training for the current model took around 12 hours
# Training was performed with a different model architecture aswell, to find the best suitable architecture for this task
# All training information can be found in appendix section of the conversation log
# In[1]:
import os
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten, Dense
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import matplotlib.pyplot as plt
from importlib import reload

# In[2]:
# Defining the 11 questions and 37 answers, therefore there are 37 different classes for this challenge
classes = [
    'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1',
    'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3',
    'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3',
    'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6',
    'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2',
    'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
    'Class11.5', 'Class11.6']

# Function to add a .jpg extension to a given filename (all images provided by GalaxyZoo are in jpg format)
# This function is used just to quickly add .jpg extension to all the filenames in the id column inside the
# given .csv file with all the solutions, used for the training
def add_extension(filename):
    return filename + ".jpg"

# Reading the csv data using the pandas library
csv_data = pd.read_csv('training_solutions_rev1.csv')
# Adding the .jpg extension to all ids in the csv
# By doing this, the data generator can easily associate the given images with the solutions from csv dataframe
csv_data["id"] = csv_data['GalaxyID'].astype(str).apply(add_extension)

# Creating a keras ImageDataGernerator to feed the training and validation images to the training process
# GalaxyZoo provides us around 60,000 images, with all the solutions for them
# With this data generator, validation_split is set to 0.1 (10%), meaning that around 54,000 images will be used 
# for training, and rest 6000 of them for validation (it doesnt matter which ones will be used for validation as 
# the galaxy zoo challenge gives us solutions for all the images)
# Keras training input generators have been examined using following tutorial:
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

data_generator = ImageDataGenerator(
    fill_mode='nearest',
    cval=0,
    rescale=1. / 255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

# Creating the batch generator for the training dataset
# Batch size has been tweaked to not stress out the GPU too much, it was first tried with batch size of 64
# and the GPU was getting exhausted because of large dataset, it was then reduced to 32 and was working fine
# The batch generator was fed with the images, and the csv solutions data
# All the photos are also getting resized and downsampled to 224x224 
training_batch_generator = data_generator.flow_from_dataframe(
    dataframe=csv_data,
    directory="data/training",
    x_col="id",
    y_col=classes,
    subset="training",
    batch_size=32,
    seed=11,
    shuffle=True,
    class_mode="raw",
    target_size=(224, 224))

# Creating the batch generator for the validation dataset
# The batch size for this generator was set to 32 aswell, and all the rest of attributes are 
# set to be exact same as for the training batch generator
validation_batch_generator = data_generator.flow_from_dataframe(
    dataframe=csv_data,
    directory="data/training",
    x_col="id",
    y_col=classes,
    subset="validation",
    batch_size=32,
    seed=22,
    shuffle=True,
    class_mode="raw",
    target_size=(224, 224))

# Calculating step size for training and validation
training_step_size = int(training_batch_generator.n/training_batch_generator.batch_size)
validation_step_size = int(validation_batch_generator.n/validation_batch_generator.batch_size)

# In[3]:
# The final chosen model architecture, which got the best restults, was VGG16
# The architecture is quite large and generates a model file with the size of around 1.1GB
# Since the network has around 134 million trainable parameters, it was chosen to import the pre-trained
# VGG16 model from the keras library and fine-tune it for this specific task to speed up the process a bit

# Defining the VGG16 model architecture, with input shape (224x224x3)
# We are also not including the top (output) layers because we need to replace them with our own containing 37 outputs
# VGG16 architecture was examined and implemented by this example https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

vgg_model = VGG16(include_top = False, input_shape=(224, 224, 3))

# Adding our own output layers for this specific task
# Adding the flattening, 2 fully connected layers and the output layer with 37 outputs and sigmoid curve as the activation function
x = Flatten(name='flatten')(vgg_model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(len(classes), activation='sigmoid', name='predictions')(x)

# Creating the final model by connecting the vgg16 model as the input and the created output model as the output
model = Model(inputs=vgg_model.input, outputs=x)
# model.summary() is used many times for diagnostics and to check how the layers in the architecture are stacked
model.summary()

# Enabling the model fine-tuning by setting all the layers to be trainable
for layer in model.layers:
    layer.trainable = True

# After some tweaking, the best results were achieved using the RMS optimiser with learning rate 1e-4
# Learning rates le-6 and le-2 were also tried, but the current one gave the best results
# Adam optimiser has also been tried with another model architecture, but RMS was still giving the best results
# Adam optimiser results with the ResNet50 network can be found in the appendix section of conversation log
optimizer = RMSprop(lr=1e-4)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


# In[4]:
# Creating the Loss history tracker, and examining how do keras callbacks work by following the next tutorial:
# https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

# Defining early stopping, which stops the training process after there was not any   
# improvement after in this case, 7 epochs
early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, verbose=1, mode='auto')

history = LossHistory()
# Creating a keras model checkpointer which saves the best model to tmp folder 
# save_best_only has been set to true, mainly because my training Azure NV6 machine had
# 30GB of SSD storage and the chosen VGG16 architecture ended up giving the model file sizes with around 1.1GB
checkpointer = ModelCheckpoint(
    filepath='tmp/weights-new.hdf5', verbose=2, save_best_only=True)

# In[5]:
# Using the keras fit_generator to train the model
# Epoch size was set to 50 but it never reached anything above 35 in all the tests I've done due to the EarlyStopper
# detecting that there's no improvement
# All the callbacks are tracked to the LossHistory object so a graph could be sketchen in the end to show
# how the training process performed
history = model.fit_generator(
    training_batch_generator,
    steps_per_epoch=training_step_size,
    validation_data=validation_batch_generator,
    validation_steps=validation_step_size,
    epochs=50,
    verbose=2,
    callbacks=[history, checkpointer, early_stopping])


# In[6]:
# Using matplotlib to sketch the loss and validation loss values and show the 
# CNN learning curve
plt.figure(figsize=(12, 10))
plt.plot(history.epoch, history.history['loss'], label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()

# In[7]:
# Even though the best model is already saved to the tmp folder, still saving the model manually just in case
model.save('trained_model.h5')