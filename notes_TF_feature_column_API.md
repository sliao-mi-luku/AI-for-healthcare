# TensorFlow Feature Column API

This document describes:

1. How to use **TensorFlow Feature Column API** to train a model on selected features
2. How to use **Aequitas** to study demographic biases and fairness of the model
3. How to **estimate the uncertainty** of the model

References:

1. Udacity's AI for Healthcare Nanodegree
2. https://www.tensorflow.org/tutorials/keras/regression
3. https://github.com/dssg/aequitas/blob/master/docs/source/examples/compas_demo.ipynb
4. https://www.tensorflow.org/probability
5. https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html
6. https://github.com/kweinmeister/notebooks/blob/master/tensorflow-shap-college-debt.ipynb
7. https://towardsdatascience.com/understand-how-your-tensorflow-model-is-making-predictions-d0b3c7e88500

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
# Preprocess
df = df['CATE_COL'].replace({1: cate1, 2:cate2, ...})

# create the vocab file
# a vocab is a csv file where each row is a unique value of a categorical variable
# put a OOV placeholder in the first row
vacob_file = unique_vocab_list.to_csv(...)

# create categorical feature from a file (for large cardinality)
cat_feature = tf.feature_column.categorical_column_with_vocabulary_file(key=COL_NAME,
                                                                        vocabulary_file=VOCAB_PATH,
                                                                        num_oov_buckets=1)
# or, (for small cardinality)
cat_feature = tf.feature_column.categorical_column_with_vocabulary_list(key=COL_NAME,
                                                                        vocabulary_list=VOCAB_LIST,
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

## Cross column
```python

cross_age_gender_feature = tf.feature_column.crossed_column([bucket_age_feature,
                                                             gender_vocab_list,
                                                             hash_bucket_size=1000])

cross_age_gender_feature = tf.feature_column.indicator_column(cross_age_gender_feature)
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



```python
dense_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```


## Null value
```python
# number of rows with at least a single null value
sum(df.apply(lambda x: sum(x.isnull().values), axis=1) > 0)

# number of null values in a single column
len(df[df['col'].isnull()])
```

## Example

We have dataframe with column `gender` and `age`, we want to build a model to predict `trestbps` (the resting blood pressure). The `gender` column is catgorical (1 = male, 2 = female). The `age` column is numerical (ex. 17, 32, 45, ...). The `trestbps` column is a numerical as well.

```python
import tensorflow as tf
```

Given this pandas dataframe, we first convert it to a TenforFlow dataframe
```python
def df_to_tfds(df, predictor, batch_size=32):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

train_ds = df_to_tfds(df=train_df, predictor='trestbps', batch_size=32)
```

We then process the feature columns. The first column is the numeric `age` column, we convert this column into a bucketed feature
```python
from tensorflow.feature_column import numeric_column, bucketized_column

age_column = numeric_column(key='age', default_value=0, dtype=tf.float64)
age_boundaries = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_feature = bucketized_column(age_column, age_boundaries)
```

We next process the `gender` column. There are 2 categories in this column. We will use an array to list out its vocab instead of using a csv file. We use one-hot coding to preocess this column.
```python
from tensorflow.feature_column import categorical_column_with_vocabulary_list, indicator_column

gender_column = categorical_column_with_vocabulary_list(key='gender',
                                                        vocabulary_list=train_df['gender'].unique())
gender_feature = indicator_column(gender_column)
```

We create a cross feature between `age` and `gender` (to model the interaction) as a third feature. We also use one-hot coding here.
```python
from tensorflow.feature_column import crossed_column

age_gender_column = crossed_column([age_feature, gender_column], hash_bucket_size=1000)
age_gender_feature = indicator_column(age_gender_column)
```

We now have 3 features: `age`, `gender`, and `age_gender`:
```python
features = [age_gender_feature, age_feature, gender_feature]
```

Create a dense layer to pass the features through
```python
from tensorflow.keras.layers import DenseFeatures
dense_feature_layer = DenseFeatures(features)
```

Build a small model
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

def create_model(dense_feature_layer):
    model = Sequential([dense_feature_layer,
                        Dense(64, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(1)])

    optimizer = RMSprop(0.001)

    model.complile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model

model = create_model(dense_feature_layer)
```

Set the early stopping
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(monitor='mse', patience=50)
```

Train the model
```python
NUM_EPOCHS = 1000

history = model.fit(train_ds, epochs=NUM_EPOCHS, callbacks=[early_stop], verbose=1)
```

Evaluate the model
```python
loss, mae, mse = model.evaluate(test_ds, verbose=2)
```

## Demographic Bias Analysis

Aequitas - The Bias and Fairness Audit Toolkit [GitHub](https://github.com/dssg/aequitas)

```python
pip install aequitas
```

Preprocess the categorical columns, the data format should be `str`

```python
from aequitas.preprocessing import preprocess_input_df

df['CATE_COL'] = df['CATE_COL'].astype(str)

df, _ = preprocess_input_df(df)
```

Create a crosstab of the preprocessed data and calculate absolute group metrics from scores and labels
```python
from aequitas.group import Group

g = Group()
xtab, _ = g.get_crosstabs(df)
absolute_metrics = g.list_absolute_metrics(xtab)
clean_xtab = xtab.fillna(-1)
```

Visualize a single group metric

```python
from aequitas.plotting import Plot

aqp = Plot()
tpr_plot = aqp.plot_group_metric(clean_xtab, 'tpr', min_group_size=0.05)
fpr_plot = aqp.plot_group_metric(clean_xtab, 'fpr', min_group_size=0.05)
```

Evaluate biases

```python
from aequitas.bias import Bias

b = Bias()

bdf = b.get_disparity_predefined_groups(xtab,
                                        original_df=df,
                                        ref_groups_dict={'race':'Caucasian', 'sex':'Male', 'age_cat':'25 - 45'},
                                        alpha=0.05,
                                        check_significance=False)

fpr_disparity = aqp.plot_disparity(bdf, group_metric='fpr_disparity', attribute_name='race')
```

Evaluate fairness
```python
from aequitas.fairness import Fairness

f = Fairness()
fdf = f.get_group_value_fairness(bdf)
fpr_fairness = aqp.plot_fairness_group(fdf, group_metric='fpr', title=True)
```

## Uncertainty Estimate

Add a `DistributionLambda` layer into the model

```python
from tensorflow_probability.layers import DistributionalLambda
from tensorflow_probability.distributions import Normal

def create_model(feature_layer):

    dist_lambda_layer = DistributionalLambda(lambda x: Normal(loc=x[..., :1],
                                             scale=1e-3 + tf.math.softplus(0.1*x[..., 1:])))

    model = Sequential([feature_layer,
                        Dense(512, acitvation='relu'),
                        Dense(2),
                        dist_lambda_layer])

    # negative log-likelihood loss
    loss = lambda x, y: -y.log_prob(x)

    model.compile(Adam(0.05), loss=loss, metrics=['mse'])

    return model
```

## Model Interpretability with Shapley Values

- Identify biases
- Determine issues or bugs in the features
- Avoid black box models

[Shapley GitHub](https://github.com/slundberg/shap)

```python
pip install shap

import shap
shap.initjs()


def create_model(num_features):
    model = Sequential([Dense(64, activation='relu', input_shape=[num_features]),
                        Dense(64, activation='relu'),
                        Dense(1)])

    optimizer = RMSprop(1e-3)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model

# use kmeans to summarize the dataset
train_df_summary = shap.kmeans(train_df.values, 25)

explainer = shap.KernelExplainer(model.predict, train_df_summary)

shap_values = explainer.shap_values(train_df.values)

shap.summary_plot(shap_values[0], train_df)


# force plot
INSTANCE_NUM = 0
shap.force_plot(explainer.expected_value[0], shap_values[0][INSTANCE_NUM], train_df.iloc[INSTANCE_NUM,:])

NUM_ROWS = 10
shap.force_plot(explainer.expected_value[0], shap_values[0][0:NUM_ROWS], train_df.iloc[0:NUM_ROWS])
```
