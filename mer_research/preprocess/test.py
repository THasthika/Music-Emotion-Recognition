import tensorflow as tf

filenames = ["./pmemo_static.tfrecords"]
raw_dataset = tf.data.TFRecordDataset(filenames)

# Create a description of the features.
feature_description = {
    'arousal_mean': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'arousal_std': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'valence_mean': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'valence_std': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'audio': tf.io.VarLenFeature(tf.float32),
}

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)