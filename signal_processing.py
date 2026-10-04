import numpy as np
from scipy.signal import butter, filtfilt

LOW_HZ = 0.7
HIGH_HZ = 3.0

# Get the green channel, divide it by its average,
# and remove any overall rise or fall in the signal.
def _prepare_green_signal(rgb_values):

    rgb_values = np.asarray(rgb_values, dtype=np.float64)

    if rgb_values.ndim != 2 or rgb_values.shape[1] != 3:
        return None

    if len(rgb_values) < 30:
        return None

    # RGB order: Red, Green, Blue
    green = rgb_values[:, 1]

    if not np.all(np.isfinite(green)):
        return None

    mean_green = np.mean(green)

    if mean_green <= 0:
        return None

    # Divide the green values by their average.
    signal = green / mean_green

    # Find the overall rise or fall in the signal.
    x = np.arange(len(signal), dtype=np.float64)
    coefficients = np.polyfit(x, signal, 1)
    trend = np.polyval(coefficients, x)

    # Remove the overall rise or fall.
    signal = signal - trend

    return signal

# Use a Butterworth filter to keep signals
# between 0.7 and 3.0 Hz.
def _bandpass_filter(signal, sample_fps):

    if signal is None:
        return None

    if sample_fps <= 0:
        return None

    nyquist = sample_fps / 2.0

    if HIGH_HZ >= nyquist:
        return None

    low = LOW_HZ / nyquist
    high = HIGH_HZ / nyquist

    try:
        b, a = butter(
            3,
            [low, high],
            btype="band"
        )

        # filtfilt needs enough samples to work correctly.
        pad_length = 3 * max(len(a), len(b))

        if len(signal) <= pad_length:
            return None

        filtered = filtfilt(
            b,
            a,
            signal
        )

        return filtered

    except Exception:
        return None


def calculate_rppg(rgb_values, sample_fps):
    """
    Steps used to calculate the heart rate:

    Mean RGB
        ↓
    Get the green channel
        ↓
    Normalize the signal
        ↓
    Remove the overall rise or fall
        ↓
    Keep signals from 0.7 to 3.0 Hz
        ↓
    Use FFT to find frequencies
        ↓
    Find the strongest frequency
        ↓
    Convert the frequency to BPM
    """

    signal = _prepare_green_signal(rgb_values)

    if signal is None:
        return None

    filtered = _bandpass_filter(
        signal,
        sample_fps
    )

    if filtered is None:
        return None

    n = len(filtered)

    if n < 30:
        return None

    # Apply a window to reduce unwanted frequency effects.
    window = np.hanning(n)

    spectrum = np.fft.rfft(
        filtered * window
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_fps
    )

    # Calculate the strength of each frequency.
    power = np.abs(spectrum) ** 2

    # Only use frequencies between 0.7 and 3.0 Hz.
    valid = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    if not np.any(valid):
        return None

    valid_indices = np.where(valid)[0]

    # Find the frequency with the strongest signal.
    peak_index = valid_indices[
        np.argmax(power[valid])
    ]

    peak_frequency = frequencies[peak_index]

    # Convert the frequency from Hz to BPM.
    bpm = peak_frequency * 60.0

    if not np.isfinite(bpm):
        return None

    return float(bpm)


def get_rppg_waveform(rgb_values, sample_fps):

    signal = _prepare_green_signal(rgb_values)

    if signal is None:
        return None

    filtered = _bandpass_filter(
        signal,
        sample_fps
    )

    if filtered is None:
        return None

    if len(filtered) < 2:
        return None

    # Scale the signal so it is easier to display.
    maximum = np.max(np.abs(filtered))

    if maximum > 0:
        waveform = filtered / maximum
    else:
        waveform = filtered

    return waveform.astype(np.float64)