from model import build_model

from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from time import time


batch_size = 128
num_epochs = 2000
learning_rate = 0.002
momentum = 0.9

# Load data
X = np.load('audio.npy')
y = np.load('label.npy').ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=444, shuffle=True)

# Convert label to onehot
y_train = to_categorical(y_train - 1, num_classes=10)
y_test = to_categorical(y_test - 1, num_classes=10)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

model = build_model(lr=learning_rate, momentum=momentum)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
          verbose=1, callbacks=[tensorboard])
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
model.save('audio_model.h5')

print('*********************')
print('Test score:', score)
print('Test accuracy:', acc)
print('*********************')