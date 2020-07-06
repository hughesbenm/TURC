from keras.datasets import mnist
import numpy as np
import pandas as pds
import keras
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import UpSampling3D
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt

# Get a batch of two random images and show in a pop-up window.
batch_xs, batch_ys = mnist.test.next_batch(2)
def preImg(img, net):
    gen_image(img).show()
    print(net.predict(img))
def img(img):
    gen_image(img).show()


y_training = np.ones(60000)
y_testing = np.concatenate((np.ones((10000)), np.zeros(2000)), axis=0)


dis = Sequential()
gen = Sequential()
GAN = Sequential()


# gen.add(Dense(196, input_shape=(100,), activation='relu'))
# gen.add(Reshape((14, 14,)))
# gen.add(UpSampling2D(size=(2, 2)))
# gen.summary()
dim = 7
depth = 256
dropout = 0.6


gen.add(Dense(784, input_dim=100))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Activation('relu'))
gen.add(Reshape((28, 28)))
gen.add(Dropout(dropout))
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
#gen.add(UpSampling2D())
gen.add(BatchNormalization(momentum=0.9))
gen.add(Activation('relu'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Activation('relu'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Activation('relu'))
#gen.add(Reshape((28, 28)))
gen.summary()


gen.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

fake = gen.predict(np.random.uniform(-1, 1, (10000, 100)))

x_testing = np.random.uniform(0, 255, (2000, 28, 28))
x_testing = np.concatenate((x_test, x_testing), axis=0)

x_train[0].shape


y_training = np.concatenate((y_training, np.zeros(60000)), axis=0)
y_training.shape

x_training = np.random.randint(0, 255, (50000, 28, 28))


x_training = np.concatenate((x_train, x_training), axis=0)
x_training = np.concatenate((x_training, fake), axis=0)
x_training.shape

dis.add(Dropout(0.6, input_shape=(28, 28,)))
dis.add(Flatten(input_shape=(28, 28,)))
dis.add(Dense((784)))
dis.add(LeakyReLU(alpha=0.2))
dis.add(Dense(392))
dis.add(LeakyReLU(alpha=0.2))
dis.add(Dense(1, activation='sigmoid'))
dis.summary()

dis.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='adam')
dis.fit(x_training, y_training, batch_size=250, epochs=20, verbose=1, validation_data=(x_testing, y_testing))

test = np.random.uniform(0, 255, (1, 28, 28))
preImg(test, dis)
i = 0
j = 0
while i < 10:
    if y_train[j] == i:
        preImg(x_train[j:j+1], dis)
        i += 1
    j += 1

GAN.add(gen)

dis.trainable = False
GAN.add(dis)

GAN.summary()
GAN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
pred = gen.predict(np.random.uniform(-1, 1, (1, 100)))
gen_image(pred).show()

for i in range(10000):
    noise = np.random.uniform(-1, 1, (15, 100))
    ones = np.ones((15, 1))
    preImg(gen.predict(noise[0:1]), dis)
    loss = GAN.train_on_batch(noise, ones)
    print(loss)
    print("Epoch: " + str(i))
#

GAN.predict(np.random.uniform(-1, 1, (1, 100)))


pred = gen.predict(np.random.uniform(-1, 1, (1, 100)))
preImg(pred, dis)
print(pred)


dis.predict(np.random.uniform(0, 255, (1, 28, 28)))


img = gen.predict(np.random.uniform(-1, 1, (1, 100)))


pred = gen.predict(np.random.uniform(-1, 1, (1, 100)))
preImg(pred, dis)

