"""
Relevant Papers:
https://arxiv.org/pdf/1706.07156v1.pdf
https://arxiv.org/pdf/1608.04363.pdf
https://arxiv.org/pdf/1703.06902v1.pdf
http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
"""

import numpy as np
import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from time import time

# Load data
X = np.load('audio.npy')
y = np.load('label.npy').ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=444, shuffle=True)

# Create the model
model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(193, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.002, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Convert label to onehot
y_train = keras.utils.to_categorical(y_train - 1, num_classes=10)
y_test = keras.utils.to_categorical(y_test - 1, num_classes=10)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model.fit(X_train, y_train, batch_size=128, epochs=2000, verbose=1, callbacks=[tensorboard])
score, acc = model.evaluate(X_test, y_test, batch_size=128)

model.save('audio_model.h5')

print('Test score:', score)
print('Test accuracy:', acc)
