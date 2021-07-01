# A_2DConv_Cat

## version 1
- nnAudio (STFT) -> 2DConv -> Linear

- n_fft
- win_length = n_fft
- freq_bins = n_fft // 2 + 1
- hop_size = n_fft // 4

## version 2
- nnAudio (MelSpec) -> 2DConv -> Linear

## version 3
- nnAudio (STFT + MelSpec + MFCC) -> 2DConv -> Linear

