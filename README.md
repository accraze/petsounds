# petsounds

Audio Classification of Wildlife Sounds using Deep Learning.

This works by training a Convolutional Neural Network trained on a subset of the [ESC-50](https://github.com/karoldvl/ESC-50) dataset. We use various spectral features extracted from the audio files and then train on the different categories of sounds. Finally, we can perform classification with a small amount of data taken from field recordings from SW Washington State.

Current Test accuracy: 0.88749

Read more in the blog post: https://accraze.info/wildlife-audio-analysis-and-classification-pt-1/

<img src="https://github.com/accraze/petsounds/blob/master/imgs/spectral-frogs.png?raw=true"/>

## Dependencies

```
keras
librosa
numpy
soundfile
sklearn
```

## Usage

### Training

First extract the audio features:

```
$ python extract.py
```

Next, train the convolutional neural network:

```
$ python main.py
```
