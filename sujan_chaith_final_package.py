import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, butter, filtfilt, welch
import tkinter as tk
from tkinter import simpledialog

# Function to generate a noisy ECG-like signal
def generate_noisy_ecg_signal(t, noise_amplitude):
    ecg_signal = np.sin(t) + 0.5 * np.sin(2 * t)
    noise = noise_amplitude * np.random.randn(len(t))
    return ecg_signal + noise

# Function to apply a Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Function to detect QRS complexes
def detect_qrs(signal, fs):
    peaks, _ = find_peaks(signal, distance=int(0.2 * fs))
    return peaks

# Function to calculate heart rate from QRS complexes
def calculate_heart_rate(peaks, fs):
    rr_intervals = np.diff(peaks) / fs
    heart_rate = 60 / rr_intervals
    return heart_rate

# Function to plot the Power Spectral Density (PSD)
def plot_psd(signal, fs, title):
    f, Pxx = welch(signal, fs, nperseg=1024)
    plt.figure()
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.grid(True)
    plt.show()

# Function to perform Hilbert transform on EEG signal
def hilbert_transform_eeg(signal):
    return hilbert(signal)

# Function to perform Hilbert transform on EMG signal
def hilbert_transform_emg(signal):
    return hilbert(signal)

# Function to extract features from EEG signal
def extract_features(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    max_value = np.max(signal)
    return [mean, variance, max_value]

# Function to get input from the user using Tkinter for ECG signal
def get_user_input_ecg():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    duration = simpledialog.askfloat("Input", "Enter duration of the ECG signal (in seconds): ")
    fs = simpledialog.askinteger("Input", "Enter sampling frequency (in Hz): ")
    p_wave_frequency = simpledialog.askfloat("Input", "Enter frequency of the P wave (in Hz): ")
    qrs_frequency = simpledialog.askfloat("Input", "Enter frequency of the QRS complex (in Hz): ")
    t_wave_frequency = simpledialog.askfloat("Input", "Enter frequency of the T wave (in Hz): ")
    noise_amplitude = simpledialog.askfloat("Input", "Enter amplitude of Gaussian noise: ")

    return duration, fs, p_wave_frequency, qrs_frequency, t_wave_frequency, noise_amplitude

# Function to get input from the user using Tkinter for EEG signal
def get_user_input_eeg():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    duration = simpledialog.askfloat("Input", "Enter duration of the EEG signal (in seconds): ")
    fs = simpledialog.askinteger("Input", "Enter sampling frequency (in Hz): ")
    # Add more EEG-specific input prompts here if needed

    return duration, fs

# Function to get input from the user using Tkinter for EMG signal
def get_user_input_emg():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    duration = simpledialog.askfloat("Input", "Enter duration of the EMG signal (in seconds): ")
    fs = simpledialog.askinteger("Input", "Enter sampling frequency (in Hz): ")
    # Add more EMG-specific input prompts here if needed

    return duration, fs

# Function to train a simple classifier
def train_classifier(features, labels):
    # Dummy classifier example: classify based on mean feature value
    threshold = np.mean(features)
    predictions = [1 if f > threshold else 0 for f in features]
    return predictions

# Function to evaluate classifier accuracy
def evaluate_classifier(predictions, labels):
    accuracy = np.mean(predictions == labels)
    print("Accuracy:", accuracy)

# Main function to generate ECG signal and plot results
def generate_ecg_signal(duration, fs, p_wave_frequency, qrs_frequency, t_wave_frequency, noise_amplitude):
    # Generate time points for the signal
    t = np.linspace(0, duration, int(fs * duration))

    # Generate a noisy ECG signal
    noisy_ecg_signal = generate_noisy_ecg_signal(t, noise_amplitude)

    # Apply a bandpass filter to remove noise
    filtered_ecg_signal = butter_bandpass_filter(noisy_ecg_signal, 0.5, 40, fs)

    # Perform the Hilbert transform (not used for ECG)
    analytic_signal = hilbert(filtered_ecg_signal)
    amplitude_envelope = np.abs(analytic_signal)

    # Detect QRS complexes
    peaks = detect_qrs(filtered_ecg_signal, fs)

    # Calculate heart rate
    heart_rate = calculate_heart_rate(peaks, fs)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_ecg_signal, label='Noisy ECG Signal', color='blue')
    plt.title('Original Signal with Noise')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_ecg_signal, label='Filtered ECG Signal', color='green')
    plt.plot(peaks / fs, filtered_ecg_signal[peaks], 'ro', label='Detected QRS Complexes')
    plt.title('Filtered Signal with Detected QRS Complexes')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    # Annotate detected QRS complexes
    for peak in peaks:
        plt.annotate('QRS', xy=(peak / fs, filtered_ecg_signal[peak]), xytext=(peak / fs, filtered_ecg_signal[peak] + 0.1),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.subplot(3, 1, 3)
    plt.plot(t, amplitude_envelope, label='Amplitude Envelope', color='red')
    plt.title('Amplitude Envelope (Hilbert Transform)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the heart rate over time
    plt.figure(figsize=(10, 4))
    plt.plot(peaks[:-1] / fs, heart_rate, label='Heart Rate', color='purple')
    plt.title('Heart Rate')
    plt.xlabel('Time [s]')
    plt.ylabel('Heart Rate [bpm]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the Power Spectral Density (PSD) of the original noisy ECG signal
    plot_psd(noisy_ecg_signal, fs, 'Power Spectral Density (Original Noisy ECG Signal)')

# Main function to generate EEG signal and plot results
def generate_eeg_signal(duration, fs):
    # Generate time points for the signal
    t = np.linspace(0, duration, int(fs * duration))

    # Generate EEG signal (example)
    eeg_signal = np.sin(2 * np.pi * 10 * t)  # Example EEG signal

    # Perform Hilbert transform on EEG signal
    analytic_signal = hilbert_transform_eeg(eeg_signal)
    amplitude_envelope = np.abs(analytic_signal)

    # Plot EEG signal and its envelope
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, eeg_signal, color='green')
    plt.title('EEG Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, amplitude_envelope, color='blue')
    plt.title('Amplitude Envelope (Hilbert Transform)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Perform prediction
    generate_eeg_signal_with_prediction(duration, fs)

# Main function to generate EMG signal and plot results
def generate_emg_signal(duration, fs):
    # Generate time points for the signal
    t = np.linspace(0, duration, int(fs * duration))

    # Generate EMG signal (example)
    emg_signal = np.random.randn(len(t))  # Example EMG signal

    # Perform Hilbert transform on EMG signal
    analytic_signal = hilbert_transform_emg(emg_signal)
    amplitude_envelope = np.abs(analytic_signal)

    # Plot EMG signal and its envelope
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, emg_signal, color='red')
    plt.title('EMG Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, amplitude_envelope, color='blue')
    plt.title('Amplitude Envelope (Hilbert Transform)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Perform prediction
    generate_emg_signal_with_prediction(duration, fs)

# Main function to generate EEG signal, extract features, and perform prediction
def generate_eeg_signal_with_prediction(duration, fs):
    # Generate time points for the signal
    t = np.linspace(0, duration, int(fs * duration))

    # Generate EEG signal (example)
    eeg_signal = np.sin(2 * np.pi * 10 * t)  # Example EEG signal
    
    # Extract features from the EEG signal
    features = extract_features(eeg_signal)
    
    # Generate labels (example: 0 for normal, 1 for abnormal)
    labels = np.random.randint(0, 2, size=len(features))
    
    # Train the classifier
    predictions = train_classifier(features, labels)
    
    # Evaluate the classifier
    evaluate_classifier(predictions, labels)
    
    # Perform Hilbert transform on EEG signal
    analytic_signal = hilbert_transform_eeg(eeg_signal)
    amplitude_envelope = np.abs(analytic_signal)

    # Plot EEG signal and its envelope
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, eeg_signal, color='green')
    plt.title('EEG Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, amplitude_envelope, color='blue')
    plt.title('Amplitude Envelope (Hilbert Transform)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Main function to generate EMG signal, extract features, and perform prediction
def generate_emg_signal_with_prediction(duration, fs):
    # Generate time points for the signal
    t = np.linspace(0, duration, int(fs * duration))

    # Generate EMG signal (example)
    emg_signal = np.random.randn(len(t))  # Example EMG signal
    
    # Extract features from the EMG signal
    features = extract_features(emg_signal)
    
    # Generate labels (example: 0 for normal, 1 for abnormal)
    labels = np.random.randint(0, 2, size=len(features))
    
    # Train the classifier
    predictions = train_classifier(features, labels)
    
    # Evaluate the classifier
    evaluate_classifier(predictions, labels)
    
    # Perform Hilbert transform on EMG signal
    analytic_signal = hilbert_transform_emg(emg_signal)
    amplitude_envelope = np.abs(analytic_signal)

    # Plot EMG signal and its envelope
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, emg_signal, color='red')
    plt.title('EMG Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, amplitude_envelope, color='blue')
    plt.title('Amplitude Envelope (Hilbert Transform)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Get input from the user using Tkinter for ECG signal
duration_ecg, fs_ecg, p_wave_frequency_ecg, qrs_frequency_ecg, t_wave_frequency_ecg, noise_amplitude_ecg = get_user_input_ecg()

# Generate and plot the ECG signal based on user input
generate_ecg_signal(duration_ecg, fs_ecg, p_wave_frequency_ecg, qrs_frequency_ecg, t_wave_frequency_ecg, noise_amplitude_ecg)

# Get input from the user using Tkinter for EEG signal
duration_eeg, fs_eeg = get_user_input_eeg()

# Generate and plot the EEG signal based on user input
generate_eeg_signal(duration_eeg, fs_eeg)

# Get input from the user using Tkinter for EMG signal
duration_emg, fs_emg = get_user_input_emg()

# Generate and plot the EMG signal based on user input
generate_emg_signal(duration_emg, fs_emg)
