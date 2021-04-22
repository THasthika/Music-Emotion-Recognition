from os import path

import torchaudio

from .base import BaseAudioDataset

class DeamDataset(BaseAudioDataset):

    """
    label_type = (categorical|static|dynamic)
    """

    def __init__(self,
                 audio_dir,
                 meta_file,
                 label_type="categorical",
                 sample_rate=22050,
                 duration=30):
        super().__init__(meta_file, sample_rate, duration)
        self.audio_dir = audio_dir
        self.label_type = label_type

    def get_labels(self):
        if self.label_type == "categorical":
            return self.meta['quadrants']
        if self.label_type == "static":
            return self.meta[['static_valence_mean', 'static_valence_std', 'static_arousal_mean', 'static_arousal_std']]
        if self.label_type == "dynamic":
            return self.meta[['dynamic_valence_mean', 'dynamic_valence_std', 'dynamic_arousal_mean', 'dynamic_arousal_std']]
        raise NameError

    def get_label(self, index):
        item = self.meta.iloc[index]
        if self.label_type == "categorical":
            label = item['quadrant']
            return label
        if self.label_type == "static":
            ret_cols = ['static_valence_mean', 'static_valence_std', 'static_arousal_mean', 'static_arousal_std']
            return torch.tensor(item[ret_cols])
        if self.label_type == "dynamic":
            ret_cols = ['dynamic_valence_mean', 'dynamic_valence_std', 'dynamic_arousal_mean', 'dynamic_arousal_std']
            return torch.tensor(item[ret_cols])
        raise NameError

    def get_audio(self, index):
        item = self.meta.iloc[index]
        audio_file = path.join(self.audio_dir, "{}.mp3".format(item['song_id']))
        x, sr = torchaudio.load(audio_file)
        return (x, sr)