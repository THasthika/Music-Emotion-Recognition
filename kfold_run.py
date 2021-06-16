from execs.a_1dconv import audio_1dconv_kfold_run

audio_1dconv_kfold_run(up_model_config={
    'raw_audio_extractor_units': [
        1, 2048, 1024, 512
    ]
})