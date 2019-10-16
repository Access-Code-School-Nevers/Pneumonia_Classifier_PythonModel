import numpy as np 
import cv2


from model import *
from get_data import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_folder = 'chest_xray/train'
test_folder = 'chest_xray/test'
valid_folder = 'chest_xray/val'

labels = {'helthy' : 0, 'bacteria' : 1, 'virus' : 2}

path_train, y_train = get_data(train_folder)
path_valid, y_valid = get_data(valid_folder)

path_train = np.array(path_train)
y_train = np.array(y_train)

path_valid = np.array(path_valid)
y_valid = np.array(y_valid)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    rescale=1/255,
    )


# randomly permutate the link, in order to shuffle data
permutation_train = np.random.permutation(path_train.shape[0])
path_train = path_train[permutation_train]
y_train = y_train[permutation_train]

# randomly permutate the link, in order to shuffle data
permutation_valid = np.random.permutation(path_valid.shape[0])
path_valid = path_valid[permutation_valid]
y_valid = y_valid[permutation_valid]


x_split = np.array_split(path_train,10)
y_split = np.array_split(y_train,10)

model = get_model()
model.summary()

for (split_x, split_y) in zip(x_split, y_split):
    # path to image (train data)
    # load images
    x_train = []
    y_train = split_y
    idx = 1
    for x in split_x:
        img = cv2.imread(x)
        img = cv2.resize(img, (640,480))
        x_train.append(img)
        idx += 1
        if idx % 100 ==0:
            print('{} images loaded'.format(idx))

    x_train = np.array(x_train)

    datagen.fit(x_train)

    batch_size = 32
    epochs = 10

    model.fit_generator(
        datagen.flow(x_train,y_train, batch_size=batch_size)
        ,steps_per_epoch=x_train.shape[0]/batch_size,
         epochs=epochs, 
         shuffle=True,
         verbose=1)

