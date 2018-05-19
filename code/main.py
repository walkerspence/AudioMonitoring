import pyaudio
import audioop
import aubio
import time
import scipy.io.wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as func_animation
from scipy.fftpack import fft

#RECORDING SETUP
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 30
SAVE = False
WAVE_OUTPUT_FILENAME = "../out/file.wav"
audio = pyaudio.PyAudio()

#PITCH TRACK SETUP
tolerance = 0.8
win_s = 4096  # fft size
bin_s = CHUNK  # bin size
pitch_out = aubio.pitch("default", win_s, bin_s, RATE)
pitch_out.set_unit("midi")
pitch_out.set_tolerance(tolerance)

def get_data_from_chunk(stream, chunk, dtype=np.int16):
    """
    :param dtype: data type (default np.float32, should be an np datadype)
    :param chunk: size of chunk to be read (int)
    :param stream: audio stream to read from (generated with pyaudio's open method)
    :return: data: data in integer format
    """
    data = np.fromstring(stream.read(chunk, exception_on_overflow=False), dtype=dtype)
    return data


# def get_fft_from_data(data, chunk):
#     """
#     :param data: data to calculate FFT of
#     :param chunk: since of chunks of data
#     :return: returns fft bins and abs value fft decomposition of bins
#     """
#     data_fft = scipy.signal.stft(data, chunk)
#     print(len(data_fft[2]))
#     return data_fft[1], abs(data_fft[2][0].real)

def get_pitch_from_data(data):
    """
    :param data: sound data in integer format
    :return: pitch, confidence
    """
    data = data.astype(np.float32)
    pitch = pitch_out(data)[0]
    confidence = pitch_out.get_confidence()
    return pitch, confidence


def monitor_mic_volume(rate, chunk, channels, live_stream, length_recording, save=False):
    """
    :param rate: rate in hz (frames/sec)
    :param chunk: size of buffer (frames)
    :param channels: number of chanels (1 = mono, 2 = stereo)
    :param live_stream: stream object (generated with pyaudio's open method)
    :param length_recording: length in seconds of recording
    :param save: true if you want to save/output audio (defaults to False)
    :return a numpy array of audio output
    """
    saved_frames = []

    for _ in range(0, rate // chunk * length_recording):
        data = get_data_from_chunk(live_stream, chunk)
        pitch, confidence = get_pitch_from_data(data)
        if save:
            saved_frames = saved_frames + [data]

        rms = audioop.rms(data, channels)
        print(pitch, rms)

    if save:
        np_data = np.hstack(saved_frames)
        print(str(length_recording) + " seconds saved")
        return np_data

    return None


def main():
    live_stream = audio.open(format=pyaudio.paInt16,
                             rate=RATE,
                             channels=CHANNELS,
                             frames_per_buffer=CHUNK,
                             input=True)

    output = monitor_mic_volume(RATE, CHUNK, CHANNELS, live_stream, RECORD_SECONDS, save=SAVE)

    if output is not None:
        scipy.io.wavfile.write(WAVE_OUTPUT_FILENAME, RATE, output)

    live_stream.stop_stream()
    live_stream.close()
    audio.terminate()

    print(str(RECORD_SECONDS) + " seconds monitored")


if __name__ == '__main__':
    main()
    exit(0)
