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

TRAIN_ROWS = 20000

PREDICT_ROWS = 80


gen = Sequential()
dis = Sequential()
GAN = Sequential()


#This is Differences
x_train = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', usecols=[17, 29], nrows=TRAIN_ROWS)

#This is Exact Values
# x_train = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', usecols=[1, 2, 4, 5, 17, 18, 19, 20, 21, 22], nrows=20000)
x_train.head()

# plt.hist(x_train['START_HOUR'], density=True, bins=50)
# plt.ylabel('Freq')

# x_hist = ["" for x in range(20000)]

# for i in range(20000):
#     if x_train['DIFFERENCE'][i] == 0:
#             x_hist[i] = "0"
#     for j in range(404):
#         if x_train['DIFFERENCE'][i] > j*60 and x_train['DIFFERENCE'][i] <= (j+1)*60:
#             x_hist[i] = (j+1)

# plt.hist(x_hist, density=True, bins=50)
# plt.ylabel('Freq')

# max = 1
# for i in range(20000):
#     if x_train['DIFFERENCE'][i] > max:
#         max = x_train['DIFFERENCE'][i]
# print(max)

y_train = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', usecols=[0], nrows=20000)
y_train.head()


# x_predict = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', usecols=[17, 28], nrows=PREDICT_ROWS)
# x_predict.head()
# x_predict = np.ones((1, 10))
# x_predict[0] = [-84.32392, -84.32392, 34.0364, 34.0364, 0, 0, 5, 21, 56, 7]


#Differences
x_test = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', skiprows=220000, usecols=[17, 28], nrows=2500)

#Exact
# x_test = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', skiprows=220000, usecols=[1, 2, 4, 5, 17, 18, 19, 20, 21, 22], nrows=2500)
x_test.head()


y_test = pds.read_excel(r'C:\Users\Ben\Desktop\Data\DifferencesData.xlsx', usecols=[0], skiprows=220000, nrows=2500)
y_test.head()


gen.add(Dense(75, input_shape=(200,), activation='relu'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Dropout(0.4))
gen.add(Dense(50, activation='relu'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Dense(25, activation='relu'))
gen.add(BatchNormalization(momentum=0.9))
gen.add(Dense(2, activation='relu'))
gen.summary()


gen.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


fake = gen.predict(np.random.uniform(-1, 1, (2500, 200)))
print(fake) 


x_training = np.ones((2500, 2))
for i in range(2500):
#     x_training[i][0] = np.random.randint(0, 23, (1, 1))
#     x_training[i][1] = np.random.randint(0, 23, (1, 1))
#     x_training[i][2] = np.random.randint(0, 59, (1, 1))
#     x_training[i][3] = np.random.randint(0, 59, (1, 1))
#     x_training[i][4] = np.random.randint(0, 59, (1, 1))
#     x_training[i][5] = np.random.randint(0, 59, (1, 1))

#     x_training[i][0] = np.random.randint(24, 100, (1, 1))
#     x_training[i][1] = np.random.randint(24, 100, (1, 1))
#     x_training[i][2] = np.random.randint(60, 100, (1, 1))
#     x_training[i][3] = np.random.randint(60, 100, (1, 1))
#     x_training[i][4] = np.random.randint(60, 100, (1, 1))
#     x_training[i][5] = np.random.randint(60, 100, (1, 1))
    
    x_training[i][0] = np.random.randint(0, 24, (1, 1))
    x_training[i][1] = np.random.randint(3600, 4800, (1, 1))
print(x_training)
x_training = np.concatenate((x_train, x_training))
print(x_training)
x_training = np.concatenate((x_training, fake))
print(x_training)


x_training = np.concatenate((x_training, np.random.uniform(-1, 1, (5000, 2))))
x_training = np.concatenate((x_training, np.zeros((1000, 2))))
x_training


y_training = np.concatenate((y_train, np.zeros((11000, 1))))
y_training


x_testing = np.concatenate((x_test, np.random.uniform(-1, 1, (2500, 2))), axis=0)
x_testing


y_testing = np.concatenate((y_test, np.zeros((2500, 1))), axis=0)
y_testing


GAN.add(gen)


dis.add(Dropout(0.4, input_shape=(2,)))
dis.add(Dense(10, activation='relu'))
# dis.add(LeakyReLU(alpha=0.2))
dis.add(BatchNormalization(momentum=0.9))
dis.add(Dense(5, activation='relu'))
# dis.add(LeakyReLU(alpha=0.2))
dis.add(BatchNormalization(momentum=0.9))
dis.add(Dense(1, activation='sigmoid'))
dis.summary(0)


print(x_training.shape)
print(y_training.shape)


# import keras
# from keras.callbacks import TensorBoard
# cb = TensorBoard(log_dir='./Discriminate/DealWithIt', histogram_freq=0, write_graph=True, write_images=True)
dis.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dis.fit(x_training, y_training, batch_size=1000, epochs=10000, verbose=1, validation_data=(x_testing, y_testing))


# pred = dis.predict(x_predict)
# print(pred)


dis.trainable = False


GAN.add(dis)


GAN.summary()
GAN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


gen.predict(np.random.uniform(-1, 1, (1, 200)))


for i in range(1000):
    noise = np.random.uniform(-1, 1, (32, 200))
    ones = np.ones((32, 1))
    print("Predicted Data:")
    print(gen.predict(noise[0:1]))
    print(dis.predict(gen.predict(noise[0:1])))
    print("\n")
    loss = GAN.train_on_batch(noise, ones)
    print(loss)
    print("Epoch: " + str(i))


random = np.random.uniform(-1, 1, (1, 200))
print(random)
data = gen.predict(random)
print(data)
pred = dis.predict(data)
print(pred)
