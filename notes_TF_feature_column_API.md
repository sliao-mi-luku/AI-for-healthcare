# TensorFlow Feature Column API

This document describes how to use TF Feature Column API

References: Udacity AI for Healthcare Nanodegree

```python
import tensorflow as tf
```

## Load data directly into TF

```python
tf_data = tf.data.experimental.make_csv_dataset(file_pattern=FILE_PATH,
                                                batch_size=BATCH_SIZE,
                                                label_name=PREDICTOR_FIELD,
                                                num_epochs=1)
```

## Convert a dataframe into tf dataset
```python
# adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns
df = pd.read_csv("...")

def df_to_dataset(df, predictor, batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = td.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

ds = df_to_dataset(...)
ds_batch = next(iter(ds))[0]
```

## Create TF numeric feature

```python
tf_feature = tf.feature_column.numeric_column(key='age',
                                              default_value=0,
                                              dtype=tf.float64)
```


## Buckets TF numeric feature
```python
bucket_boundaries = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
tf_bucket_feature = tf.feature_column.bucketized_column(source_column=tf_feature,
                                                        boundaries=bucket_boundaries)
```

## Categorical feature
```python
# create the vocab file
# a vocab is a csv file where each row is a unique value of a categorical variable
# put a OOV placeholder in the first row
vacob_file = unique_vocab_list.to_csv(...)

# create categorical feature
cat_feature = tf.feature_column.categorical_column_with_vocabulary_file(key=COL_NAME,
                                                                        vocabulary_file=VOCAB_PATH,
                                                                        num_oov_buckets=1)
# create one-hot encoding feature
one_hot_feature = tf.feature_column.indicator_column(cat_feature)

# create embedding feature
embedding_feature = tf.feature_column.embedding_column(categorical_column,
                                                       dimension,
                                                       combiner='mean',
                                                       initializer=None,
                                                       ckpt_to_load_from=None,
                                                       tensor_name_in_ckpt=None,
                                                       max_norm=None,
                                                       trainable=True,
                                                       use_safe_embedding_lookup=True)
```


## Example
```python
import functools

def normalize(col, mean, std):

    return (col-mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_val=0):

    normalizer = functools.partial(normalize, mean=MEAN, std=STD)

    return tf.feature_column.numeric_column(key=col,
                                            default_value=deault_val,
                                            normalizer_fn=normalizer,
                                            dtype=tf.float64)

example_tf_feature = create_tf_numeric_feature('age', MEAN, STD)

feature_layer = tf.keras.layers.DenseFeatures(example_tf_feature)

```

## DenseFeatures

https://www.tensorflow.org/tutorials/keras/regression

```python
dense_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```
