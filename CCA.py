import numpy as np
from sklearn.cross_decomposition import CCA

def generate_reference_signals(f, n_samples, fs, n_harmonics=3):
    t = np.arange(n_samples) / fs
    ref = []
    for i in range(1, n_harmonics + 1):
        ref.append(np.sin(2 * np.pi * f * i * t))
        ref.append(np.cos(2 * np.pi * f * i * t))
    return np.array(ref).T  # shape: (n_samples, 2 * n_harmonics)

def ssvep_cca_classify(eeg_trial, freqs, fs=256, n_harmonics=3):
    n_samples = eeg_trial.shape[1]
    cca = CCA(n_components=1)

    corrs = []
    for f in freqs:
        Yf = generate_reference_signals(f, n_samples, fs, n_harmonics)
        cca.fit(eeg_trial.T, Yf)
        U, V = cca.transform(eeg_trial.T, Yf)
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        corrs.append(corr)

    predicted_idx = np.argmax(corrs)
    return freqs[predicted_idx]

# Preload your data
from scipy.io import loadmat
mat = loadmat("EEGData/s1.mat")
eeg = mat["eeg"]  # (12 targets, 8 channels, 1114 samples, 15 trials)

stimulus_freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                  10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

# Example
target_idx = 0
trial_idx = 0
stim_onset = 39
trial = eeg[target_idx, :, stim_onset:, trial_idx]
trial = (trial - np.mean(trial, axis=1, keepdims=True)) / np.std(trial, axis=1, keepdims=True)

predicted = ssvep_cca_classify(trial, stimulus_freqs)
print(f"Predicted: {predicted} Hz | True: {stimulus_freqs[target_idx]} Hz")

correct = 0
total = 0

for target_idx in range(12):
    for trial_idx in range(15):
        trial = eeg[target_idx, :, stim_onset:, trial_idx]
        trial = (trial - trial.mean(axis=1, keepdims=True)) / trial.std(axis=1, keepdims=True)
        predicted = ssvep_cca_classify(trial, stimulus_freqs)
        true = stimulus_freqs[target_idx]
        correct += (predicted == true)
        total += 1

print(f"Accuracy: {correct}/{total} = {100 * correct / total:.2f}%")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = []
y_pred = []

for target_idx in range(12):
    for trial_idx in range(15):
        trial = eeg[target_idx, :, stim_onset:, trial_idx]
        trial = (trial - trial.mean(axis=1, keepdims=True)) / trial.std(axis=1, keepdims=True)
        predicted = ssvep_cca_classify(trial, stimulus_freqs)
        y_pred.append(stimulus_freqs.index(predicted))
        y_true.append(target_idx)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=stimulus_freqs)
disp.plot(cmap='viridis')
plt.title("SSVEP CCA Confusion Matrix")
plt.show()
