from hole_puncher import HolePuncher
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os

from logistic_vector_regression import LogisticVectorRegression


def load_image(number):
    img_dir = '../data/64x64'
    path = os.path.join(img_dir, f"{number:05d}.png")
    return np.array(ndimage.imread(path, flatten=False)) / 255.0


TRAIN_SIZE = 2500
TEST_SIZE = 5
IMG_SHAPE = (64, 64, 3)

images = []
for i in range(TRAIN_SIZE):
    images.append(load_image(i+1))


puncher = HolePuncher(IMG_SHAPE)

X_train, Y_train = puncher.split_batch(images)


with LogisticVectorRegression() as reg:

    reg.fit(X_train, Y_train,
            training_epochs=200,
            learning_rate=0.01)

    for i in range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE):

        image_test = load_image(i+1)
        x_test, y_test = puncher.split(image_test)
        y_hat = reg.predict(x_test[:, np.newaxis])[:, 0]
        merged = puncher.merge(x_test, y_hat)

        plt.imshow(merged)
        plt.show()






