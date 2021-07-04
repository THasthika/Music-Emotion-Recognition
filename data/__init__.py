import torch
import torchaudio

SONG_ID = 'song_id'
START_TIME = 'start_time'
END_TIME = 'end_time'
CLIP_START_TIME = 'clip_start_time'
DURATION = 'duration'

QUADRANT = 'quadrant'
STATIC_VALENCE_MEAN = 'static_valence_mean'
STATIC_VALENCE_STD = 'static_valence_std'
STATIC_AROUSAL_MEAN = 'static_arousal_mean'
STATIC_AROUSAL_STD = 'static_arousal_std'
DYNAMIC_VALENCE_MEAN = 'dynamic_valence_mean'
DYNAMIC_VALENCE_STD = 'dynamic_valence_std'
DYNAMIC_AROUSAL_MEAN = 'dynamic_arousal_mean'
DYNAMIC_AROUSAL_STD = 'dynamic_arousal_std'

def preprocess_audio(frame_count, audio, sr, ret_sr):
    x = torch.mean(audio, 0, True)
    out = torch.zeros(1, frame_count)
    effects = [
        ["rate", f"{ret_sr}"]
    ]
    x, sr2 = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
    if frame_count >= x.shape[1]:
        out[:, :x.shape[1]] = x
    else:
        out[:, :] = x[:, :frame_count]
    # out = torch.squeeze(out)
    # out = torch.unsqueeze(out, dim=1)
    return out