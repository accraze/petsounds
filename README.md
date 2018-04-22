# petsounds

Audio Classification of Animal Sounds using Deep Learning.

Current Test accuracy: 0.88749
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