from models import BaseStatModel
import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

from nnAudio import Spectrogram

device = "cuda" if torch.cuda.is_available() else "cpu"

class ACL1DConvLSTMStat_V1(BaseStatModel):

    AUDIO_HIDDEN_SIZE = "audio_hidden_size"
    AUDIO_NUM_LAYERS = "audio_num_layers"

    STFT_HIDDEN_SIZE = "stft_hidden_size"
    STFT_NUM_LAYERS = "stft_num_layers"

    MEL_SPEC_HIDDEN_SIZE = "mel_spec_hidden_size"
    MEL_SPEC_NUM_LAYERS = "mel_spec_num_layers"

    MFCC_HIDDEN_SIZE = "mfcc_hidden_size"
    MFCC_NUM_LAYERS = "mfcc_num_layers"

    N_FFT = "n_fft"
    N_MELS = "n_mels"
    N_MFCC = "n_mfcc"
    SPEC_TRAINABLE = "spec_trainable"

    def __init__(self,
                batch_size=32,
                num_workers=4,
                train_ds=None,
                val_ds=None,
                test_ds=None,
                **model_config):
        super().__init__(batch_size, num_workers, train_ds, val_ds, test_ds, **model_config)

        self.__build_model()
    
    def __build_model(self):

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert_model = self.bert_model.to(device)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        f_bins = (self.config[self.N_FFT] // 2) + 1

        self.stft = Spectrogram.STFT(n_fft=self.config[self.N_FFT], fmax=9000, sr=22050, trainable=self.config[self.SPEC_TRAINABLE], output_format="Magnitude")
        self.mel_spec = Spectrogram.MelSpectrogram(sr=22050, n_fft=self.config[self.N_FFT], n_mels=self.config[self.N_MELS], trainable_mel=self.config[self.SPEC_TRAINABLE], trainable_STFT=self.config[self.SPEC_TRAINABLE])
        self.mfcc = Spectrogram.MFCC(sr=22050, n_mfcc=self.config[self.N_MFCC])

        self.audio_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=250, kernel_size=1024, stride=256),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1),
            nn.BatchNorm1d(250),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU()
        )

        self.audio_lstm = nn.LSTM(
            input_size=250,
            hidden_size=self.config[self.AUDIO_HIDDEN_SIZE],
            num_layers=self.config[self.AUDIO_NUM_LAYERS]
        )

        self.stft_feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=f_bins, out_channels=500, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),

            nn.Conv1d(in_channels=500, out_channels=500, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU()

        )

        self.stft_lstm = nn.LSTM(
            input_size=500,
            hidden_size=self.config[self.STFT_HIDDEN_SIZE],
            num_layers=self.config[self.STFT_NUM_LAYERS]
        )

        self.mel_spec_feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=self.config[self.N_MELS], out_channels=100, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(),

            nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU()

        )


        self.mel_spec_lstm = nn.LSTM(
            input_size=100,
            hidden_size=self.config[self.MEL_SPEC_HIDDEN_SIZE],
            num_layers=self.config[self.MEL_SPEC_NUM_LAYERS]
        )

        self.mfcc_feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=self.config[self.N_MFCC], out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()

        )

        self.mfcc_lstm = nn.LSTM(
            input_size=32,
            hidden_size=self.config[self.MFCC_HIDDEN_SIZE],
            num_layers=self.config[self.MFCC_NUM_LAYERS]
        )

        self.lyrics_extractor = nn.LSTM(input_size=768, hidden_size=250, num_layers=1)

        input_size = self.config[self.AUDIO_HIDDEN_SIZE]
        input_size += self.config[self.STFT_HIDDEN_SIZE]
        input_size += self.config[self.MEL_SPEC_HIDDEN_SIZE]
        input_size += self.config[self.MFCC_HIDDEN_SIZE]
        input_size += 250

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(in_features=128, out_features=2)
        )

        self.fc_std = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            self._get_std_activation()
        )

    def forward(self, x):

        (audio_x, lyrics_x) = x

        a_x = self.audio_feature_extractor(audio_x)

        stft_x = self.stft(audio_x)
        stft_x = self.stft_feature_extractor(stft_x)

        mel_x = self.mel_spec(audio_x)
        mel_x = self.mel_spec_feature_extractor(mel_x)

        mfcc_x = self.mfcc(audio_x)
        mfcc_x = self.mfcc_feature_extractor(mfcc_x)

        a_x = a_x.permute((0, 2, 1))
        stft_x = stft_x.permute((0, 2, 1))
        mel_x = mel_x.permute((0, 2, 1))
        mfcc_x = mfcc_x.permute((0, 2, 1))

        (out, _) = self.audio_lstm(a_x)
        a_x = out[:, -1, :]

        (out, _) = self.stft_lstm(stft_x)
        stft_x = out[:, -1, :]

        (out, _) = self.mel_spec_lstm(mel_x)
        mel_x = out[:, -1, :]

        (out, _) = self.mfcc_lstm(mfcc_x)
        mfcc_x = out[:, -1, :]

        lyrics_x = self.bert_tokenizer(lyrics_x, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False, return_attention_mask=False)['input_ids']
        lyrics_x = lyrics_x.to(device)
        with torch.no_grad():
            lyrics_x = self.bert_model(lyrics_x)
        lyrics_x = lyrics_x[0]
        (lyrics_x, _) = self.lyrics_extractor(lyrics_x)
        lyrics_x = lyrics_x[:, -1, :]

        x = torch.cat((a_x, stft_x, mel_x, mfcc_x, lyrics_x), dim=1)

        x = self.fc(x)
        x_mean = self.fc_mean(x)
        x_std = self.fc_std(x)
        x = torch.cat((x_mean, x_std), dim=1)
        return x
