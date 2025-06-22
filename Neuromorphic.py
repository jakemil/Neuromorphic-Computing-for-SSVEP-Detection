import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.signal import hilbert, butter, filtfilt, welch
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class AdaptiveSpikingNeuron:
    """Enhanced LIF neuron with adaptive threshold and refractory period"""

    def __init__(self, tau_m=20.0, v_thresh=1.0, v_reset=0.0, dt=1.0,
                 refractory_period=5, adaptation_strength=0.1):
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.dt = dt
        self.refractory_period = refractory_period
        self.adaptation_strength = adaptation_strength

        self.v_mem = 0.0
        self.spike_times = []
        self.refractory_counter = 0
        self.adaptive_thresh = v_thresh
        self.last_spike_time = -float('inf')

    def step(self, input_current, time_step):
        """Enhanced LIF dynamics with adaptation"""
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return 0.0

        # Adaptive threshold decay
        time_since_spike = time_step - self.last_spike_time
        self.adaptive_thresh = self.v_thresh + (self.adaptive_thresh - self.v_thresh) * np.exp(
            -time_since_spike / (self.tau_m * 2))

        # Leaky integration with noise
        noise = np.random.normal(0, 0.01)
        self.v_mem += (-self.v_mem + input_current + noise) * self.dt / self.tau_m

        # Check for spike
        if self.v_mem >= self.adaptive_thresh:
            self.spike_times.append(time_step)
            self.v_mem = self.v_reset
            self.refractory_counter = self.refractory_period
            self.last_spike_time = time_step
            # Increase adaptive threshold
            self.adaptive_thresh += self.adaptation_strength
            return 1.0
        return 0.0

    def reset(self):
        self.v_mem = 0.0
        self.spike_times = []
        self.refractory_counter = 0
        self.adaptive_thresh = self.v_thresh
        self.last_spike_time = -float('inf')


class EnhancedTemporalCodingFilter:
    """Improved neuromorphic temporal coding with multiple encoding strategies"""

    def __init__(self, target_freq, fs, n_neurons=20, tau_range=(5, 100),
                 encoding_type='mixed'):
        self.target_freq = target_freq
        self.fs = fs
        self.n_neurons = n_neurons
        self.encoding_type = encoding_type

        # Create diverse neuron populations
        taus = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_neurons // 2)
        thresholds = np.linspace(0.5, 1.5, n_neurons // 2)

        self.neurons = []
        # Population 1: Varied time constants
        for tau in taus:
            self.neurons.append(AdaptiveSpikingNeuron(tau_m=tau))
        # Population 2: Varied thresholds
        for thresh in thresholds:
            self.neurons.append(AdaptiveSpikingNeuron(v_thresh=thresh))

        # Enhanced phase detection
        self.n_phase_bins = 8
        self.phase_detectors = np.linspace(0, 2 * np.pi, self.n_phase_bins, endpoint=False)

    def encode_signal_rate(self, signal):
        """Rate-based encoding"""
        spike_trains = []
        for neuron in self.neurons:
            neuron.reset()
            spikes = []
            for t, sample in enumerate(signal):
                # Rectify and scale input
                input_current = np.clip(sample * 3.0, -2.0, 4.0)
                spike = neuron.step(input_current, t)
                spikes.append(spike)
            spike_trains.append(spikes)
        return np.array(spike_trains)

    def encode_signal_temporal(self, signal):
        """Temporal pattern encoding"""
        # Use Hilbert transform for phase information
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.angle(analytic_signal)
        instantaneous_freq = np.diff(np.unwrap(instantaneous_phase)) * self.fs / (2 * np.pi)

        spike_trains = []
        for i, neuron in enumerate(self.neurons):
            neuron.reset()
            spikes = []
            preferred_phase = self.phase_detectors[i % len(self.phase_detectors)]

            for t, (sample, phase) in enumerate(zip(signal, instantaneous_phase)):
                # Phase-dependent activation
                phase_diff = np.abs(np.angle(np.exp(1j * (phase - preferred_phase))))
                activation = np.exp(-phase_diff ** 2 / 0.5) * np.abs(sample) * 2.0
                spike = neuron.step(activation, t)
                spikes.append(spike)
            spike_trains.append(spikes)
        return np.array(spike_trains)

    def encode_signal(self, signal):
        """Mixed encoding strategy"""
        if self.encoding_type == 'rate':
            return self.encode_signal_rate(signal)
        elif self.encoding_type == 'temporal':
            return self.encode_signal_temporal(signal)
        else:  # mixed
            rate_trains = self.encode_signal_rate(signal)
            temporal_trains = self.encode_signal_temporal(signal)
            return np.vstack([rate_trains[:len(self.neurons) // 2],
                              temporal_trains[:len(self.neurons) // 2]])

    def compute_enhanced_phase_locking(self, spike_trains, signal):
        """Enhanced phase locking with multiple metrics"""
        if len(signal) == 0:
            return 0.0

        t = np.arange(len(signal)) / self.fs
        analytic_signal = hilbert(signal)
        instantaneous_phase = np.angle(analytic_signal)

        # Target frequency phase
        target_phase = 2 * np.pi * self.target_freq * t

        phase_scores = []
        spike_timing_scores = []

        for spike_train in spike_trains:
            spike_indices = np.where(np.array(spike_train) > 0)[0]
            if len(spike_indices) < 2:
                continue

            # Phase locking to target frequency
            phases_at_spikes = target_phase[spike_indices]
            if len(phases_at_spikes) > 0:
                cos_sum = np.sum(np.cos(phases_at_spikes))
                sin_sum = np.sum(np.sin(phases_at_spikes))
                vector_strength = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / len(phases_at_spikes)
                phase_scores.append(vector_strength)

            # Inter-spike interval regularity
            if len(spike_indices) > 1:
                isis = np.diff(spike_indices) / self.fs
                expected_isi = 1.0 / self.target_freq
                isi_regularity = np.exp(-np.mean(np.abs(isis - expected_isi)) / expected_isi)
                spike_timing_scores.append(isi_regularity)

        phase_score = np.mean(phase_scores) if phase_scores else 0.0
        timing_score = np.mean(spike_timing_scores) if spike_timing_scores else 0.0

        return 0.7 * phase_score + 0.3 * timing_score


class AdvancedAdaptiveDetector:
    """Multi-scale adaptive threshold with outlier detection"""

    def __init__(self, adaptation_rates=[0.05, 0.1, 0.2], window_sizes=[50, 100, 200]):
        self.adaptation_rates = adaptation_rates
        self.window_sizes = window_sizes
        self.stats = []

        for rate, window in zip(adaptation_rates, window_sizes):
            self.stats.append({
                'rate': rate,
                'window': window,
                'running_mean': 0,
                'running_std': 1,
                'history': []
            })

    def update_threshold(self, signal):
        """Multi-scale threshold adaptation"""
        signal_flat = signal.flatten()

        for stat in self.stats:
            stat['history'].extend(signal_flat)
            if len(stat['history']) > stat['window']:
                stat['history'] = stat['history'][-stat['window']:]

            if len(stat['history']) > 10:
                # Robust statistics
                current_median = np.median(stat['history'])
                current_mad = np.median(np.abs(stat['history'] - current_median))
                current_std = current_mad * 1.4826  # Convert MAD to std

                stat['running_mean'] = ((1 - stat['rate']) * stat['running_mean'] +
                                        stat['rate'] * current_median)
                stat['running_std'] = ((1 - stat['rate']) * stat['running_std'] +
                                       stat['rate'] * current_std)

    def normalize_signal(self, signal):
        """Multi-scale normalization"""
        normalized_signals = []

        for stat in self.stats:
            norm_signal = (signal - stat['running_mean']) / (stat['running_std'] + 1e-8)
            normalized_signals.append(norm_signal)

        # Weighted combination (favor shorter windows for responsiveness)
        weights = np.array([1.0, 0.7, 0.4])
        weights = weights / np.sum(weights)

        final_signal = np.zeros_like(signal)
        for i, norm_sig in enumerate(normalized_signals):
            final_signal += weights[i] * norm_sig

        return final_signal


class EnhancedNeuromorphicSSVEPClassifier:
    """Significantly improved neuromorphic SSVEP classifier"""

    def __init__(self, freqs, fs=256, n_harmonics=4, enable_adaptation=True,
                 n_neurons=30, ensemble_size=3):
        self.freqs = freqs
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.enable_adaptation = enable_adaptation
        self.ensemble_size = ensemble_size

        # Create ensemble of neuromorphic filters
        self.temporal_filter_ensembles = {}
        for freq in freqs:
            self.temporal_filter_ensembles[freq] = []
            for i in range(ensemble_size):
                encoding_types = ['rate', 'temporal', 'mixed']
                filter_obj = EnhancedTemporalCodingFilter(
                    freq, fs, n_neurons=n_neurons,
                    encoding_type=encoding_types[i % len(encoding_types)]
                )
                self.temporal_filter_ensembles[freq].append(filter_obj)

        if enable_adaptation:
            self.adaptive_detector = AdvancedAdaptiveDetector()

        # Enhanced CCA
        self.cca = CCA(n_components=2)  # Use more components

        # Frequency-specific preprocessing
        self.freq_filters = {}
        for freq in freqs:
            self.freq_filters[freq] = self._design_filter(freq)

    def _design_filter(self, freq):
        """Design frequency-specific bandpass filter"""
        nyquist = self.fs / 2
        # Wider bandwidth for better phase information
        bandwidth = 3.0
        low = max(0.5, (freq - bandwidth)) / nyquist
        high = min(0.99, (freq + bandwidth)) / nyquist

        try:
            b, a = butter(6, [low, high], btype='band')  # Higher order
            return (b, a)
        except:
            return None

    def extract_spectral_features(self, signal, freq):
        """Extract frequency-domain features"""
        freqs_psd, psd = welch(signal, self.fs, nperseg=min(256, len(signal) // 4))

        # Find peak at target frequency and harmonics
        target_power = 0
        for harmonic in range(1, self.n_harmonics + 1):
            target_freq = freq * harmonic
            freq_idx = np.argmin(np.abs(freqs_psd - target_freq))
            if freq_idx < len(psd):
                target_power += np.mean(psd[:, freq_idx - 1:freq_idx + 2], axis=1)

        # Normalize by total power
        total_power = np.sum(psd, axis=1)
        snr = target_power / (total_power + 1e-8)

        return {
            'snr': np.mean(snr),
            'peak_power': np.mean(target_power),
            'power_ratio': np.mean(target_power) / (np.mean(total_power) + 1e-8)
        }

    def extract_neuromorphic_features(self, eeg_trial):
        """Enhanced neuromorphic feature extraction"""
        if self.enable_adaptation:
            self.adaptive_detector.update_threshold(eeg_trial)
            eeg_normalized = self.adaptive_detector.normalize_signal(eeg_trial)
        else:
            eeg_normalized = eeg_trial

        features = {}

        for freq in self.freqs:
            freq_features = {
                'phase_locking': [],
                'channel_consistency': [],
                'max_channel': [],
                'spectral_features': [],
                'ensemble_agreement': []
            }

            ensemble_predictions = []

            # Process each channel
            for ch_idx in range(eeg_normalized.shape[0]):
                channel_signal = eeg_normalized[ch_idx, :]

                # Apply frequency-specific filtering
                if self.freq_filters[freq] is not None:
                    b, a = self.freq_filters[freq]
                    try:
                        channel_signal = filtfilt(b, a, channel_signal)
                    except:
                        pass

                # Extract spectral features
                spectral_feat = self.extract_spectral_features(
                    channel_signal[np.newaxis, :], freq)
                freq_features['spectral_features'].append(spectral_feat)

                # Ensemble of neuromorphic filters
                ensemble_scores = []
                for temporal_filter in self.temporal_filter_ensembles[freq]:
                    spike_trains = temporal_filter.encode_signal(channel_signal)
                    phase_score = temporal_filter.compute_enhanced_phase_locking(
                        spike_trains, channel_signal)
                    ensemble_scores.append(phase_score)

                ensemble_predictions.append(ensemble_scores)
                freq_features['phase_locking'].append(np.mean(ensemble_scores))
                freq_features['max_channel'].append(np.max(ensemble_scores))

            # Aggregate features
            phase_scores = freq_features['phase_locking']
            freq_features['channel_consistency'] = 1.0 / (1.0 + np.std(phase_scores))
            freq_features['ensemble_agreement'] = np.mean([
                1.0 / (1.0 + np.std(scores)) for scores in ensemble_predictions
            ])

            # Combine spectral features
            spectral_combined = {
                'snr': np.mean([sf['snr'] for sf in freq_features['spectral_features']]),
                'peak_power': np.mean([sf['peak_power'] for sf in freq_features['spectral_features']]),
                'power_ratio': np.mean([sf['power_ratio'] for sf in freq_features['spectral_features']])
            }

            features[freq] = {
                'phase_locking': np.mean(phase_scores),
                'channel_consistency': freq_features['channel_consistency'],
                'max_channel': np.mean(freq_features['max_channel']),
                'ensemble_agreement': freq_features['ensemble_agreement'],
                'spectral_snr': spectral_combined['snr'],
                'spectral_power': spectral_combined['peak_power']
            }

        return features

    def compute_enhanced_cca(self, eeg_trial, freq):
        """Enhanced CCA with improved reference signals"""
        if self.freq_filters[freq] is not None:
            b, a = self.freq_filters[freq]
            try:
                filtered_eeg = filtfilt(b, a, eeg_trial, axis=1)
            except:
                filtered_eeg = eeg_trial
        else:
            filtered_eeg = eeg_trial

        n_samples = eeg_trial.shape[1]
        t = np.arange(n_samples) / self.fs
        ref = []

        # Enhanced reference signals with multiple harmonics and phases
        for i in range(1, self.n_harmonics + 1):
            for phase in [0, np.pi / 2]:  # Sin and cos for each harmonic
                ref.append(np.sin(2 * np.pi * freq * i * t + phase))

        Yf = np.array(ref).T

        try:
            self.cca.fit(filtered_eeg.T, Yf)
            U, V = self.cca.transform(filtered_eeg.T, Yf)

            # Use multiple canonical components
            corr_scores = []
            for comp in range(min(2, U.shape[1], V.shape[1])):
                corr = np.corrcoef(U[:, comp], V[:, comp])[0, 1]
                if not np.isnan(corr):
                    corr_scores.append(abs(corr))

            return np.mean(corr_scores) if corr_scores else 0.0
        except:
            return 0.0

    def classify(self, eeg_trial, use_fusion=True):
        """Enhanced classification with improved fusion"""
        neuro_features = self.extract_neuromorphic_features(eeg_trial)

        cca_scores = {}
        for freq in self.freqs:
            cca_scores[freq] = self.compute_enhanced_cca(eeg_trial, freq)

        if use_fusion:
            combined_scores = {}
            for freq in self.freqs:
                # Enhanced neuromorphic score
                neuro_score = (
                        0.4 * neuro_features[freq]['phase_locking'] +
                        0.2 * neuro_features[freq]['channel_consistency'] +
                        0.2 * neuro_features[freq]['ensemble_agreement'] +
                        0.1 * neuro_features[freq]['spectral_snr'] +
                        0.1 * neuro_features[freq]['max_channel']
                )

                cca_score = cca_scores[freq]

                # Adaptive fusion weights based on confidence
                neuro_confidence = neuro_features[freq]['ensemble_agreement']
                cca_confidence = min(1.0, 2.0 * cca_score)  # Scale CCA confidence

                total_confidence = neuro_confidence + cca_confidence + 1e-8
                neuro_weight = 0.4 + 0.4 * (neuro_confidence / total_confidence)
                cca_weight = 1.0 - neuro_weight

                combined_scores[freq] = neuro_weight * neuro_score + cca_weight * cca_score

            best_freq = max(combined_scores, key=combined_scores.get)
        else:
            # Enhanced neuromorphic-only scoring
            neuro_scores = {}
            for freq in self.freqs:
                neuro_scores[freq] = (
                        0.5 * neuro_features[freq]['phase_locking'] +
                        0.3 * neuro_features[freq]['ensemble_agreement'] +
                        0.2 * neuro_features[freq]['spectral_snr']
                )
            best_freq = max(neuro_scores, key=neuro_scores.get)

        return best_freq


def enhanced_preprocess_eeg(eeg_data, fs=256):
    """Enhanced preprocessing pipeline"""
    processed = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[0]):
        channel_data = eeg_data[ch, :]

        # Remove DC offset
        channel_data = channel_data - np.mean(channel_data)

        # Robust outlier removal
        q75, q25 = np.percentile(channel_data, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        channel_data = np.clip(channel_data, lower_bound, upper_bound)

        # Bandpass filter (1-50 Hz)
        nyquist = fs / 2
        try:
            b, a = butter(4, [1.0 / nyquist, 50.0 / nyquist], btype='band')
            channel_data = filtfilt(b, a, channel_data)
        except:
            pass

        # Robust standardization
        median = np.median(channel_data)
        mad = np.median(np.abs(channel_data - median))
        processed[ch, :] = (channel_data - median) / (mad * 1.4826 + 1e-8)

    return processed


def test_enhanced_classifier():
    """Test the enhanced neuromorphic classifier"""
    from scipy.io import loadmat

    try:
        mat = loadmat("EEGData/s1.mat")
        eeg = mat["eeg"]
    except:
        print("Data file not found. Please ensure EEGData/s1.mat exists.")
        return None, 0, 0

    stimulus_freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                      10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

    # Initialize enhanced classifier
    neuro_classifier = EnhancedNeuromorphicSSVEPClassifier(
        stimulus_freqs, fs=256, enable_adaptation=True,
        n_neurons=30, ensemble_size=3
    )

    stim_onset = 39
    correct_fusion = 0
    correct_neuro_only = 0
    total = 0

    y_true = []
    y_pred_fusion = []
    y_pred_neuro = []

    print("Testing Enhanced Neuromorphic SSVEP Classifier...")
    print("=" * 60)

    for target_idx in range(12):
        target_correct_fusion = 0
        target_correct_neuro = 0

        for trial_idx in range(15):
            trial = eeg[target_idx, :, stim_onset:, trial_idx]
            trial = enhanced_preprocess_eeg(trial)

            predicted_fusion = neuro_classifier.classify(trial, use_fusion=True)
            predicted_neuro = neuro_classifier.classify(trial, use_fusion=False)

            true_freq = stimulus_freqs[target_idx]

            if predicted_fusion == true_freq:
                correct_fusion += 1
                target_correct_fusion += 1
            if predicted_neuro == true_freq:
                correct_neuro_only += 1
                target_correct_neuro += 1

            total += 1

            y_true.append(target_idx)
            y_pred_fusion.append(stimulus_freqs.index(predicted_fusion))
            y_pred_neuro.append(stimulus_freqs.index(predicted_neuro))

        print(f"Target {target_idx + 1:2d} ({stimulus_freqs[target_idx]:5.2f} Hz): "
              f"Fusion {target_correct_fusion:2d}/15 ({100 * target_correct_fusion / 15:5.1f}%), "
              f"Neuro {target_correct_neuro:2d}/15 ({100 * target_correct_neuro / 15:5.1f}%)")

    acc_fusion = 100 * correct_fusion / total
    acc_neuro = 100 * correct_neuro_only / total

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"Enhanced Neuromorphic + CCA Fusion: {correct_fusion}/{total} = {acc_fusion:.2f}%")
    print(f"Enhanced Neuromorphic Only: {correct_neuro_only}/{total} = {acc_neuro:.2f}%")
    print(f"Improvement over original: Fusion +{acc_fusion - 63.33:.1f}%, Neuro +{acc_neuro - 20:.1f}%")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cm_fusion = confusion_matrix(y_true, y_pred_fusion)
    disp_fusion = ConfusionMatrixDisplay(cm_fusion,
                                         display_labels=[f"{f:.2f}" for f in stimulus_freqs])
    disp_fusion.plot(ax=ax1, cmap='viridis', values_format='d')
    ax1.set_title(f"Enhanced Neuromorphic Fusion\nAccuracy: {acc_fusion:.2f}%", fontsize=14)
    ax1.tick_params(axis='x', rotation=45)

    cm_neuro = confusion_matrix(y_true, y_pred_neuro)
    disp_neuro = ConfusionMatrixDisplay(cm_neuro,
                                        display_labels=[f"{f:.2f}" for f in stimulus_freqs])
    disp_neuro.plot(ax=ax2, cmap='plasma', values_format='d')
    ax2.set_title(f"Enhanced Neuromorphic Only\nAccuracy: {acc_neuro:.2f}%", fontsize=14)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return neuro_classifier, acc_fusion, acc_neuro


if __name__ == "__main__":
    classifier, fusion_acc, neuro_acc = test_enhanced_classifier()