# Amin Aliari - 9431066
# Project: Signal bouns project
# Title: 1. Voice recognition

# ----------------- imports -----------------
from os import walk

import numpy as np

from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq

import matplotlib.colors
import matplotlib.pyplot as plt
# -------------------------------------------

# ----------------- constants -----------------
MAX_MEN_FREQ = 180
MAX_WOMEN_FREQ = 255
# ---------------------------------------------


# Question 3
# get_max_freq('voices/v0.wav')
def get_max_freq(filename):
    freq, cdata = wavfile.read(filename)

    # the file is too large so i take a 10 seconds sample from it
    cdata = cdata[: 10 * freq]

    data = cdata.astype(np.float32)  # discrete data
    data = (data / np.max(np.abs(data)))
    data -= np.mean(data)

    cf = fft(data)
    positive_slice = int(len(cf) / 2) - 1
    cf = cf[:positive_slice]
    rf = abs(cf)

    freqs = fftfreq(data.shape[0], 1 / freq)
    freqs = freqs[:positive_slice]
    idx = np.argsort(freqs)

    maxf_index = np.where(rf == np.amax(rf))[0]
    maxf = freqs[maxf_index]

    return maxf


# Question 2
# plot_spectrum_graph('voices/v0.wav')
def plot_spectrum_graph(filename):
    freq, cdata = wavfile.read(filename)

    # the file is too large so i take a 10 seconds sample from it
    cdata = cdata[: 10 * freq]

    data = cdata.astype(np.float32)  # discrete data
    data = (data / np.max(np.abs(data)))
    data -= np.mean(data)

    cf = fft(data)
    positive_slice = int(len(cf) / 2) - 1
    cf = cf[:positive_slice]
    rf = abs(cf)

    freqs = fftfreq(data.shape[0], 1 / freq)
    freqs = freqs[:positive_slice]
    idx = np.argsort(freqs)

    # spectrum graph
    plt.plot(freqs[idx], rf[idx], '#0097a7')
    plt.xlabel("frequency")
    plt.ylabel("power")
    plt.show()


# Question 4
# classify('voices')
def perdict_voice_gender(spect_value) -> tuple:
    gender = "invalid"
    if (spect_value < MAX_MEN_FREQ):
        gender = "men"
    else:
        gender = "women"

    return gender, spect_value


def classify(dir) -> dict:
    ret = {}

    (_, _, files) = next(walk(dir))

    for file in files:
        path = f'{dir}/{file}'
        spect_value = get_max_freq(path)
        result = perdict_voice_gender(spect_value)
        ret[file] = result

    return ret


# main
res = classify('voices')
for key in res.keys():
    print(f'{key}: {res[key][0]}, {res[key][1]}')
