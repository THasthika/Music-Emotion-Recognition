from models.audio_1dconv import Audio1DConv

from data import AudioOnlyStaticQuadrantAndAVValues

from os import path

import kfold

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 0.01
SAMPLE_RATE = 22050
CHUNK_DURATION = 10
OVERLAP = 5

ROOT_STORAGE_FOLDER = '/storage/s3bkt/'

DATA_DIR = {
    'mer-taffc': '/raw/mer-taffc/audio/',
    'deam': '/raw/deam/audio/',
    'emomusic': '/raw/emomusic/audio/',
    'pmemo': '/raw/pmemo/audio/'
}

SPLIT_DIR = {
    'mer-taffc': '/splits/mer-taffc-kfold/',
    'deam': '/splits/deam-kfold/',
    'emomusic': '/splits/emomusic-kfold/',
    'pmemo': '/splits/pmemo-kfold/'
}

TEMP_FOLDER = '/precomputed/{}/chunked/{}-{}-{}/'

ADDITIONAL_TAGS = [
    'conv-1d',
    '{}-second-chunks'.format(CHUNK_DURATION)
]
MODEL_NAME = '104'
DATASET_NAME = 'mer-taffc'
INPUT_TYPE = 'audio' # audio | audio-lyrics | lyrics
OUTPUT_TYPE = 'emotion-class,static-av' # emotion-class | static-av | dynamic-av
FEATURES = 'raw'
DATA_CLASS = AudioOnlyStaticQuadrantAndAVValues
MODEL_CLASS = Audio1DConv


def audio_1dconv_kfold_run(n_splits=5, num_workers=4, up_model_config={}):

    model_config={
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'raw_audio_extractor_units': [1, 8, 32, 64, 128],
        'raw_audio_latent_time_units': 16,
        'classifier_units': [1024, 512, 128]
    }

    model_config.update(up_model_config)

    wandb_tags = [
        'input:{}'.format(INPUT_TYPE),
        'output:{}'.format(OUTPUT_TYPE),
        'dataset:{}'.format(DATASET_NAME),
        'model:{}'.format(MODEL_NAME)
    ]

    wandb_tags.extend(ADDITIONAL_TAGS)

    dataset_args = {
        'chunk_duration': CHUNK_DURATION,
        'overlap': OVERLAP,
        'temp_folder': TEMP_FOLDER.format(DATASET_NAME, SAMPLE_RATE, CHUNK_DURATION, OVERLAP)
    }

    model = MODEL_CLASS(
        batch_size=model_config['batch_size'],
        num_workers=4,
        config=model_config
    )

    trainer_args = {
        'max_epochs': model_config['max_epochs'],
        'checkpoint_callback': True,
        'gpus': -1
    }

    train_dataset = DATA_CLASS(
        path.join(SPLIT_DIR[DATASET_NAME], "train.json"),
        **dataset_args
    )

    test_dataset = DATA_CLASS(
        path.join(SPLIT_DIR[DATASET_NAME], "test.json"),
        **dataset_args
    )

    kfold_runner = kfold.WandBCV(
            n_splits=n_splits,
            stratify=False,
            batch_size=model_config['batch_size'],
            num_workers=num_workers,
            wandb_project_name="mer", 
            wandb_tags=wandb_tags,
            **trainer_args)

    kfold_runner.fit(model, train_dataset, test_dataset)