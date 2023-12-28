# The sounddevice library is an alternative to PyAudio that can be used for realtime/streaming data analysis

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# Print available devices
for i, device in enumerate(sd.query_devices()):
    print(f"Device {i}: {device.get('name')}")

# Manually specify which device to use
selected_device_index = 0

# Parameters for sounddevice
CHANNELS = 1
RATE = 44100
BLOCKSIZE = 1024

# Create plot
plt.ion()
fig, ax = plt.subplots()
x = np.arange(0, BLOCKSIZE)
line, = ax.plot(x, np.random.rand(BLOCKSIZE))

# Define a function to stream audio and update plot.
# This function assumes that a 'line' variable is available in the globals() array.
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    line.set_ydata(indata[:,0])
    fig.canvas.draw()
    fig.canvas.flush_events()

# Open stream and start streaming audio
stream = sd.InputStream(
    device=selected_device_index,
    channels=CHANNELS,
    samplerate=RATE,
    callback=audio_callback,
    blocksize=BLOCKSIZE
)

# Handle the stream using a contextmanager like this to ensure that it gets closed properly on exit.
with stream:  # This will call stream.start() and stream.stop() automatically.

    # Keep the plot window open
    plt.show(block=False)
    while plt.fignum_exists(1):
        plt.pause(0.1)
