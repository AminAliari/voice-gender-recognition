# Amin Aliari - 9431066
# Project: Signal bouns project
# Title: 2. Noise reduction

# ----------------- imports -----------------
from os import walk

import numpy as np
from numpy.random import seed
from numpy.random import rand

from librosa import stft, istft

from scipy.io import wavfile

import matplotlib.colors
import matplotlib.pyplot as plt
# -------------------------------------------

# ----------------- constants -----------------
NOISE_VALUE = 1
# ---------------------------------------------


def plot_graph(filename):
    freq, cdata = wavfile.read(filename)
    data = cdata.astype(np.float32)  # discrete data
    length = data.shape[0] / freq

    time = np.linspace(0, length, data.shape[0])

    plt.plot(time, data, label="wave")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.show()


def add_noise(filename) -> np.ndarray:
    freq, cdata = wavfile.read(filename)
    data = cdata.astype(np.float32)  # discrete data

    sample_count = len(data)

    seed(1)
    max_value = np.amax(data) * NOISE_VALUE
    noise = rand(sample_count) * 2 - max_value
    noise = noise.astype(np.float32)

    data = np.add(data, noise)

    wavfile.write('noisy.wav', freq, data.astype(np.int16))

    return freq, data, noise


# main
# wav_data: noisy file, noise_data = only the noise part
freq, wav_data, noise_data = add_noise("test.wav")

s = stft(wav_data)
ss = np.abs(s)
angle = np.angle(s)
b = np.exp(1.0j * angle)
ns = stft(noise_data)
nss = np.abs(ns)

mns = np.mean(nss, axis=1)
sa = ss - mns.reshape((mns.shape[0], 1))

sa0 = sa * b
y = istft(sa0)
wavfile.write('fixed.wav', freq, y.astype(np.int16))
plot_graph('fixed.wav')
