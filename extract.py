import glob
import os
import librosa
import numpy as np
import soundfile as sf


def extract_audio(file_name):
    """
    Performs necessary transformations
    for feature extraction and returns the features.
    """
    X, sample_rate = _load_sample(file_name)
    stft = _stft(X)
    mfccs = _mfccs(X, sample_rate)
    chroma = _chroma(stft, sample_rate)
    mel = _melspectrogram(X, sample_rate)
    contrast = _spectral_contrast(stft, sample_rate)
    tonnetz = _tonnetz(X,sample_rate)
    return mfccs, chroma, mel, contrast, tonnetz

def _load_sample(file_name):
    """
    Loads audio sample
    """
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T
    return X

def _stft(X):
    """
    Performs short time fourier transform.
    Returns magnitude of frequency bin f at frame t
    """
    return np.abs(librosa.stft(X))

def _mfccs(y, sample_rate):
    """
    Returns the mean Mel-frequency 
    cepstral coefficients.
    """
    return np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

def _chroma(stft, sample_rate):
    """
    Compute a chromagram from a waveform 
    or power spectrogram.

    Returns the mean normalized energy 
    for each chroma bin at each frame.
    """
    return np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)

def _melspectrogram(X, sample_rate):
    """
    Computes magnitude spectrogram and 
    then maps onto the mel scale by mel_f.dot(S**2)

    Returns the mean melspectrogram
    """
    return np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

def _spectral_contrast(stft, sample_rate):
    """
    Computes spectral contrast
    Returns the mean values that
    correspond to a given octave-based frequency
    """
    return np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)

def _tonnetz(X, sample_rate):
    """
    Computes and returns the mean tonal centroid
    features for each frame.
    """
    return np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

def parse_audio_files(parent_dir, sub_dirs, file_ext='*.ogg'):
    """
    Reads in all training audio and
    extracts the features and saves
    the audio features and labels as npy files.
    """
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_audio(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, label)
        print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype=np.int)

# Create features and labels
r = os.listdir("data/")
r.sort()
features, labels = parse_audio_files('data', r)
np.save('audio.npy', features)
np.save('label.npy', labels)
