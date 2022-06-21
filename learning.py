from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from pathlib import Path
import warnings


warnings.filterwarnings('ignore')
dir_alldata = Path('chest_xray/chest_xray')
train_data_dir = dir_alldata / 'train'
validation_data_dir = dir_alldata / 'val'
test_data_dir = dir_alldata / 'test'
pneumonia sub-directories
normal_cases_train = train_data_dir / 'NORMAL'
pneumonia_cases_train = train_data_dir / 'PNEUMONIA'
img_width, img_height = 150, 150
nb_train_sample = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 20
if K.image_data_format() == "channels_first":
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(150, 150), batch_size=batch_size,                                               class_mode="binary")
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(150, 150),                                                          batch_size=batch_size, class_mode="binary")
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(150, 150), batch_size=batch_size,                                          class_mode="binary")
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
checkpoint = ModelCheckpoint(
    'model_best_weights.h5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=1
)

history = model.fit_generator(train_generator, steps_per_epoch=nb_train_sample // batch_size, epochs=epochs,                               validation_data=validation_generator,                           validation_steps=nb_validation_samples // batch_size, callbacks=checkpoint)
test_accuracy = model.evaluate_generator(test_generator)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
