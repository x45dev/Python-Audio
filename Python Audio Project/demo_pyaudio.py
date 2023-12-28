# PyAudio can be used to process realtime/streaming input data

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

# Parameters for PyAudio
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()  # By default this will open an interface with your microphone.
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Create plot
plt.ion()
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))

# Continuously plot live audio data
while True:
    data = stream.read(CHUNK)
    data_np = np.frombuffer(data, dtype=np.float32)
    line.set_ydata(data_np)
    fig.canvas.draw()
    fig.canvas.flush_events()
