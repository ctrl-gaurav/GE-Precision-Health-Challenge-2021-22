import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os


os.listdir('../input/chest-xray-dataset-4-classes/Dataset')

IMAGE_SIZE = [224, 224]

train_path = '../input/chest-xray-dataset-4-classes/Dataset/Train'
valid_path = '../input/chest-xray-dataset-4-classes/Dataset/Val'
test_path = '../input/chest-xray-dataset-4-classes/Dataset/Test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob('../input/chest-xray-dataset-4-classes/Dataset/Train/*')
x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)




training_set = train_datagen.flow_from_directory('../input/chest-xray-dataset-4-classes/Dataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




val_set = val_datagen.flow_from_directory('../input/chest-xray-dataset-4-classes/Dataset/Val',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


test_set = test_datagen.flow_from_directory('../input/chest-xray-dataset-4-classes/Dataset/Test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


history = model.fit(
  training_set,
  validation_data=val_set,
  epochs=150
)

model.save('final.h5')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_set, batch_size=16)
print("test loss, test acc:", results)