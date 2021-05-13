def spectrogram_args(n_frames, n_frequencies, window=1024, step=None):
    if step == None:
        step = window // 2
    signal_length = window + (n_frames - 1) * step
    nfft = 2 * (n_frequencies - 1)
    return (nfft, signal_length)

def spectrogram_window(n_frames, n_frequencies, signal_length):
    nfft = 2 * (n_frequencies - 1)
    window = (2 * signal_length) // (n_frames + 1)
    return (nfft, window)

def spectrogram_dims(N, nfft, step=None):
    if step == None:
        step = nfft // 4
    n_frames = (N - window) // step + 1
    n_frequencies = nfft // 2 + 1
    return (n_frequencies, n_frames)