import numpy as np
import numpy.random as random
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import cPickle
from download import getFile

# Assumes you have an internet connection
# Download necessary files containing the CIFAR-100 dataset.
# Training set: 50K images.
# Test set: 5K images.
# Cross Validation set: 5K images.

x_test, y_test, x_validation, y_validation, x_train, y_train = getFile('testImages.npy'), getFile('testLabels.npy'), getFile('validationImages.npy'), getFile('validationLabels.npy'), getFile('trainImages.npy'), getFile('trainLabels.npy');

random.seed(1337)  # for reproducibility

imageSize = 32;
classes = 100;
batch_size = 128;
training_epochs = 100;
imageChannels = 3;

y_train = np_utils.to_categorical(y_train, classes);
y_test = np_utils.to_categorical(y_test, classes);
y_validation = np_utils.to_categorical(y_validation, classes);
x_train = x_train.astype('float32');
x_test = x_test.astype('float32');
x_validation = x_validation.astype('float32');

# NOTE: The X image data has already been normalized with a range of 0 to 1.

print("Defining sequential model");
model = Sequential();

model.add(Conv2D(182, 3, padding='same', activation='relu', data_format='channels_last', input_shape=(imageSize, imageSize, 3)));
model.add(Conv2D(182, 3, data_format='channels_last'));
model.add(BatchNormalization());
model.add(Activation('relu'));
model.add(MaxPooling2D(data_format="channels_last", pool_size=(3, 3)));
model.add(Dropout(0.45));

model.add(Conv2D(312, 3, activation='relu', data_format='channels_last'));
model.add(Conv2D(312, 3, data_format='channels_last'));
model.add(BatchNormalization());
model.add(Activation('relu'));
model.add(MaxPooling2D(data_format="channels_last", pool_size=(3, 3)));
model.add(Dropout(0.45));

model.add(Conv2D(386, 3, activation='relu', data_format='channels_last', padding='same'));
model.add(Conv2D(386, 3, data_format='channels_last', padding='same'));
model.add(BatchNormalization());
model.add(Activation('relu'));
model.add(MaxPooling2D(data_format="channels_last", pool_size=(3, 3), padding='same'));
model.add(Dropout(0.45));
model.add(Flatten());

model.add(Dense(768))
model.add(BatchNormalization());
model.add(Activation('relu'));
model.add(Dropout(0.45));

model.add(Dense(512));
model.add(BatchNormalization());
model.add(Activation('relu'));
model.add(Dropout(0.45));
model.add(Dense(classes, activation='softmax')); # Softmax forces the NN to classify. If we want to give the algorithm the ability to classify 'none', use sigmoid instead and check if any outputs are over some threshold value like 0.5.

dataGenerator = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
);

print("fitting data to generator");
dataGenerator.fit(x_train);

print("using Adam optimizer");
optimizer = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0);

print("compiling model");
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']);

print("using TensorBoard, Reduce LR On Plateau, and Early Stopping callbacks");
tensorboardCallback = keras.callbacks.TensorBoard(log_dir='./TensorBoardData', batch_size=batch_size, histogram_freq=0, write_graph=True, write_images=True);
reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0);
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto');

print("fitting data generator");
model.fit_generator(dataGenerator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=4000, epochs=training_epochs, verbose=1, validation_data=(x_validation, y_validation), callbacks=[tensorboardCallback, reduceLR, earlyStopping]);

print("evaluating model");
score = model.evaluate_generator(dataGenerator.flow(x_test, y_test, batch_size=batch_size), steps=x_test.shape[0]);

print("Training model score on test set:");
print(score);

print("Metrics used to score test: ");
print(model.metrics_names);

print('Saving model & weights to disk as cifar-weights.h5');
model.save('./cifar-weights.h5');
print('Execution complete.');
