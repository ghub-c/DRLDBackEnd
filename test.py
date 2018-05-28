# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import plot_model

#We need to split data into train, validation and test paths
train_path = './bigdataSet/training'
valid_path = './bigdataSet/validation'

#Image array is (144, 256, 3)
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size= (144, 256), classes=['street','property'], batch_size=32);
validation_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size= (144,256), classes=['street','property'], batch_size=32);

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(144, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['acc'])


plot_model(model, to_file='model2.png')

model.fit_generator(train_batches, steps_per_epoch=150, validation_data=validation_batches, epochs=50, verbose=1)

# Save model and weights to folder

#model.load_weights('./saved_models/final.h5f')
model.save_weights('./final.h5f')
