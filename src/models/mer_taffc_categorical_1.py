import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import models, layers
from tensorflow.keras.layers.experimental import preprocessing

data_dir = "./datasets/MER_taffc/wav"

SAMPLE_RATE=44100

quads = np.array(tf.io.gfile.listdir(str(data_dir)))

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label 0:', len(tf.io.gfile.listdir(os.path.join(data_dir,quads[0]))))
print('Number of examples per label 1:', len(tf.io.gfile.listdir(os.path.join(data_dir,quads[1]))))
print('Number of examples per label 2:', len(tf.io.gfile.listdir(os.path.join(data_dir,quads[2]))))
print('Number of examples per label 3:', len(tf.io.gfile.listdir(os.path.join(data_dir,quads[3]))))
print('Example file tensor:', filenames[0])

train_files = filenames[:720]
val_files = filenames[720:720+90]
test_files = filenames[-90:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# rows = 3
# cols = 3
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
# for i, (audio, label) in enumerate(waveform_ds.take(n)):
#   r = i // cols
#   c = i % cols
#   ax = axes[r][c]
#   ax.plot(audio.numpy())
#   ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
#   label = label.numpy().decode('utf-8')
#   ax.set_title(label)

# plt.show()

def get_spectrogram(waveform):
  wav_len = tf.shape(waveform)[0]
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.pad(waveform, [[0, (SAMPLE_RATE * 35) - wav_len]])
  # equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(equal_length, frame_length=512, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  return spectrogram

for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
# display(Audio(waveform, rate=SAMPLE_RATE))

# def plot_spectrogram(spectrogram, ax):
#   # Convert to frequencies to log scale and transpose so that the time is
#   # represented in the x-axis (columns).
#   log_spec = np.log(spectrogram.T)
#   height = log_spec.shape[0]
#   X = np.arange(12055)
#   Y = range(height)
#   ax.pcolormesh(X, Y, log_spec)


# fig, axes = plt.subplots(2, figsize=(12, 8))
# timescale = np.arange(waveform.shape[0])
# axes[0].plot(timescale, waveform.numpy())
# axes[0].set_title('Waveform')
# axes[0].set_xlim([0, SAMPLE_RATE*35])
# plot_spectrogram(spectrogram.numpy(), axes[1])
# axes[1].set_title('Spectrogram')
# plt.show()

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == quads)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# rows = 3
# cols = 3
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
# for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
#   r = i // cols
#   c = i % cols
#   ax = axes[r][c]
#   plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
#   ax.set_title(quads[label_id.numpy()])
#   ax.axis('off')

# plt.show()

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(quads)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(128, 128), 
    norm_layer,
    layers.Conv2D(128, 3, activation='relu'),
    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 100
history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

model.save('./models/mer_taffc_categorical_1.1.py')

# metrics = history.history
# plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
# plt.legend(['loss', 'val_loss'])
# plt.show()

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

# confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
# plt.figure(figsize=(10, 8))
# sns.heatmap(confusion_mtx, xticklabels=quads, yticklabels=commands, 
#             annot=True, fmt='g')
# plt.xlabel('Prediction')
# plt.ylabel('Label')
# plt.show()