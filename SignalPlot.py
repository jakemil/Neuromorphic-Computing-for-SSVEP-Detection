import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# --- Load EEG data ---
mat = scipy.io.loadmat('EEGData/s1.mat')
eeg = mat['eeg']  # shape: (12 targets, 8 channels, 1114 samples, 15 trials)
print('EEG shape:', eeg.shape)

# --- Metadata ---
stimulus_freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                  10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
fs = 256  # sampling rate in Hz
t = np.arange(eeg.shape[2]) / fs  # time vector (1114 samples)

# --- Choose data to visualize ---
target_idx = 0      # Index of target frequency (e.g., 0 for 9.25 Hz)
trial_idx = 0       # Trial index (0–14)

# Extract and trim the signal (remove pre-stimulus samples)
stim_onset = 39  # stimulus starts at sample 39
time = t[stim_onset:]

# --- Plot time-domain signals for all channels ---
for channel_idx in range(8):
    signal = eeg[target_idx, channel_idx, stim_onset:, trial_idx]

    plt.figure(figsize=(10, 3))
    plt.plot(time, signal)
    plt.title(f'Channel {channel_idx + 1} – {stimulus_freqs[target_idx]} Hz – Trial {trial_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
