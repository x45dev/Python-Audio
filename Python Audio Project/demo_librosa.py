"""
This file demonstrates various Librosa capabilities and visualisations
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file from the local filesystem.
filepath = r'path_to_your_audio_file.wav'  # Include 'r' prefix to interpret as a raw string. This is required for Windows paths!

# Alternatively, load from the Librosa online examples library.
# If the file hasn't been used before, it will be downloaded when this code is first run.
filepath = librosa.example('nutcracker')  # `librosa.example('<name>')` returns a string.
y, sr = librosa.load(filepath)

# ---
# Plotting the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# ---
# Plotting the spectrogram
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# ---
# Mel-frequency cepstral coefficients (MFCCs):
# MFCCs represent the short-term power spectrum of a sound.
# Extract using librosa.display.specshow
mfccs = librosa.feature.mfcc(y=y, sr=sr)

# Display MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.show()

# ---
# Chroma features
# Chromagrams represent the energy distribution of pitch classes
# Extract using librosa.display.specshow.
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

# Display chromagram
plt.figure(figsize=(10, 4))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title('Chromagram')
plt.xlabel('Time')
plt.ylabel('Pitch Class')
plt.show()

# ---
# Calculate tempo and beat frames
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# Plot beat frames
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
plt.figure(figsize=(10, 4))
plt.plot(beat_times, label='Beat frames')
plt.xlabel('Time')
plt.ylabel('Beat')
plt.title(f'Tempo: {tempo:.2f} BPM')
plt.legend()
plt.show()

# ---
# Pitch and frequency tracking
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

# Plot pitch contour
plt.figure(figsize=(10, 4))
plt.plot(librosa.times_like(pitches), pitches.T, label='Pitch (Hz)')
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch Contour')
plt.legend()
plt.show()

# ---
# Melspectrogram
# Mel-scaled spectrograms emphasize the frequency bands critical for human speech and music perception.
spectrum = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert to dB scale
spectrum_db = librosa.power_to_db(spectrum, ref=np.max)

# Display mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrum_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency (Mel)')
plt.show()
