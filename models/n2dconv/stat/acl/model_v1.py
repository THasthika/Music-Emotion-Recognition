from models import BaseStatModel

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

from nnAudio import Spectrogram

device = "cuda" if torch.cuda.is_available() else "cpu"

class ACL2DConvStat_V1(BaseStatModel):

    ADAPTIVE_LAYER_UNITS_0 = "adaptive_layer_units_0"
    ADAPTIVE_LAYER_UNITS_1 = "adaptive_layer_units_1"
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

        self.stft = Spectrogram.STFT(n_fft=self.config[self.N_FFT], fmax=9000, sr=22050,
                                     trainable=self.config[self.SPEC_TRAINABLE], output_format="Magnitude")
        self.mel_spec = Spectrogram.MelSpectrogram(sr=22050, n_fft=self.config[self.N_FFT],
                                                   n_mels=self.config[self.N_MELS],
                                                   trainable_mel=self.config[self.SPEC_TRAINABLE],
                                                   trainable_STFT=self.config[self.SPEC_TRAINABLE])
        self.mfcc = Spectrogram.MFCC(sr=22050, n_mfcc=self.config[self.N_MFCC])

        self.audio_feature_1d_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=500, kernel_size=1024, stride=256),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
        )

        self.audio_feature_2d_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 256
        input_size = (
                    self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.stft_feature_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 256
        input_size += (
                    self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.mel_spec_feature_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 256
        input_size += (
                    self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.mfcc_feature_extractor = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 8)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d(p=self.config[self.DROPOUT]),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(
                self.config[self.ADAPTIVE_LAYER_UNITS_0],
                self.config[self.ADAPTIVE_LAYER_UNITS_1]
            ))
        )

        out_channels = 64
        input_size += (
                    self.config[self.ADAPTIVE_LAYER_UNITS_0] * self.config[self.ADAPTIVE_LAYER_UNITS_1] * out_channels)

        self.lyrics_extractor = nn.LSTM(input_size=768, hidden_size=250, num_layers=1)
        self.lyrics_fc = nn.Sequential(
            nn.Linear(in_features=250, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU()
        )

        input_size += 64

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.Dropout(p=self.config[self.DROPOUT]),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
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

        raw_x = self.audio_feature_1d_extractor(audio_x)
        raw_x = torch.unsqueeze(raw_x, dim=1)
        raw_x = self.audio_feature_2d_extractor(raw_x)
        raw_x = torch.flatten(raw_x, start_dim=1)

        stft_x = self.stft(audio_x)
        stft_x = torch.unsqueeze(stft_x, dim=1)
        stft_x = self.stft_feature_extractor(stft_x)

        mel_x = self.mel_spec(audio_x)
        mel_x = torch.unsqueeze(mel_x, dim=1)
        mel_x = self.mel_spec_feature_extractor(mel_x)

        mfcc_x = self.mfcc(audio_x)
        mfcc_x = torch.unsqueeze(mfcc_x, dim=1)
        mfcc_x = self.mfcc_feature_extractor(mfcc_x)

        lyrics_x = self.bert_tokenizer(lyrics_x, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False, return_attention_mask=False)['input_ids']
        lyrics_x = lyrics_x.to(device)
        with torch.no_grad():
            lyrics_x = self.bert_model(lyrics_x)
        lyrics_x = lyrics_x[0]
        (lyrics_x, _) = self.lyrics_extractor(lyrics_x)
        lyrics_x = lyrics_x[:, -1, :]
        lyrics_x = torch.flatten(lyrics_x, start_dim=1)
        lyrics_x = self.lyrics_fc(lyrics_x)

        x = torch.cat((raw_x, stft_x, mel_x, mfcc_x, lyrics_x), dim=1)

        x = self.fc0(x)

        x_mean = self.fc_mean(x)
        x_std = self.fc_std(x)
        x = torch.cat((x_mean, x_std), dim=1)
        return x
