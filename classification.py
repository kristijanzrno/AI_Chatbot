# In[1]:
import os
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten, Dense
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import matplotlib.pyplot as plt
from importlib import reload

# In[2]:

classes = [
    'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1',
    'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3',
    'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3',
    'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6',
    'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2',
    'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
    'Class11.5', 'Class11.6']

def append_ext(fn):
    return fn + ".jpg"

csv_data = pd.read_csv('training_solutions_rev1.csv')
csv_data["id"] = csv_data['GalaxyID'].astype(str).apply(append_ext)

datagen = ImageDataGenerator(
    fill_mode='nearest',
    cval=0,
    rescale=1. / 255,
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    dataframe=csv_data,
    directory="data/training",
    x_col="id",
    y_col=classes,
    subset="training",
    batch_size=40,
    seed=11,
    shuffle=True,
    class_mode="raw",
    target_size=(224, 224))

valid_generator = datagen.flow_from_dataframe(
    dataframe=csv_data,
    directory="data/training",
    x_col="id",
    y_col=classes,
    subset="validation",
    batch_size=40,
    seed=22,
    shuffle=True,
    class_mode="raw",
    target_size=(224, 224))

training_step_size = int(train_generator.n / train_generator.batch_size)
validation_step_size = int(valid_generator.n / valid_generator.batch_size)

# In[3]:

vgg_model = VGG16(include_top = False, input_shape=(224, 224, 3))

x = Flatten(name='flatten')(vgg_model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(len(classes), activation='sigmoid', name='predictions')(x)

model = Model(inputs=vgg_model.input, outputs=x)
model.summary()

for layer in model.layers:
    layer.trainable = True

optimizer = RMSprop(lr=1e-4)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


# In[4]:
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, verbose=1, mode='auto')

history = LossHistory()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(
    filepath='tmp/weights-new.hdf5', verbose=2, save_best_only=True)

# In[5]:

history = model.fit_generator(
    train_generator,
    steps_per_epoch=training_step_size,
    validation_data=valid_generator,
    validation_steps=validation_step_size,
    epochs=50,
    verbose=2,
    callbacks=[history, checkpointer, early_stopping])


# In[6]:
plt.figure(figsize=(12, 8))
plt.plot(history.epoch, history.history['loss'], label='Training Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()

# In[7]:
model.save('trained_model.h5')