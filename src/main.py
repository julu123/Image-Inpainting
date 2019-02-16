from hole_puncher import HolePuncher
from conv_net import ConvNet
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os

from logistic_vector_regression import LogisticVectorRegression


def load_image(number):
    img_dir = '../data/64x64'
    path = os.path.join(img_dir, f"{number:05d}.png")
    return np.array(ndimage.imread(path, flatten=False)) / 255.0


TRAIN_SIZE = 1
TEST_SIZE = 5
IMG_SHAPE = (64, 64, 3)

images = []
for i in range(TRAIN_SIZE):
    images.append(load_image(i+1))


puncher = HolePuncher(IMG_SHAPE)

X_train, Y_train = puncher.split_fill_batch(images)

with ConvNet() as conv:
    conv.fit(X_train, Y_train,
             width=IMG_SHAPE[0],
             height=IMG_SHAPE[1],
             training_epochs=200,
             learning_rate=0.0001)

    test_images = []
    # for i in range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE):
    for i in range(TRAIN_SIZE):
        test_images.append(load_image(i + 1))

    X_test, Y_test = puncher.split_fill_batch(test_images)

    for i in range(X_test.shape[1]):
        x_test, y_test = X_test[:, i], Y_test[:, i]
        y_hat = conv.predict(x_test[:, np.newaxis])[:, 0]
        merged = puncher.blend(x_test, y_hat)

        plt.imshow(merged)
        plt.show()




