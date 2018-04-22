"""
CNN Model for Audio Classificiation
Uses SGD optimizer with Nesterov Momentum.

Relevant Papers:
http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf
https://arxiv.org/pdf/1706.07156v1.pdf
https://arxiv.org/pdf/1608.04363.pdf
https://arxiv.org/pdf/1703.06902v1.pdf
"""
from keras.layers import (
    Conv1D, Dense, Dropout, Embedding,
    GlobalAveragePooling1D, MaxPooling1D)
from keras.models import Sequential
from keras.optimizers import SGD


def build_model(lr=0.002, momentum=0.9):
    """
    Creates and returns CNN model
    """
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
                  optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                  metrics=['accuracy'])
    return model
