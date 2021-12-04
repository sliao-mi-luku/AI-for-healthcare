import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df = df.copy()
    
    # dict to map ndc_codes to generic names
    d = dict()
    d['nan'] = np.nan
    # create the dict
    for i in range(ndc_df.shape[0]):
        code = ndc_df.iloc[i]['NDC_Code']
        name = ndc_df.iloc[i]['Non-proprietary Name']
        d[code] = name
    
    # create an array to store generic names in df
    arr = []
    for i in range(df.shape[0]):
        code = df.iloc[i]['ndc_code']
        name = d[str(code)]
        arr.append(name)
    
    # create a new column neamed generic_drug_name
    df['generic_drug_name'] = np.array(arr)
    
    return df

#Question 4
def select_first_encounter(df):
    """
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    """
    # sort df by encounter_id column
    df = df.sort_values(by='encounter_id')
    
    # list of all first encounter ids for each patient
    first_encounter_ids = df.groupby('patient_nbr')['encounter_id'].head(1).values
    
    # select only the rows from df with first encounter ids
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_ids)]
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    # unique patient ids
    patients = df[patient_key].unique()
    # randomly shuffle the order of the patient ids
    np.random.shuffle(patients)
    # total number of patients
    n = len(patients)
    
    # train: 60%
    train_ids = patients[:int(0.6*n)]
    # validation: 20%
    val_ids = patients[int(0.6*n):int(0.8*n)]
    # test: 20%
    test_ids = patients[int(0.8*n):]
    
    train = df[df[patient_key].isin(train_ids)]
    validation = df[df[patient_key].isin(val_ids)]
    test = df[df[patient_key].isin(test_ids)]
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......
        '''
        cat_ftr_col = tf.feature_column.categorical_column_with_vocabulary_file(key=c,
                                                                                vocabulary_file=vocab_file_path,
                                                                                num_oov_buckets=1)
        cat_ftr = tf.feature_column.indicator_column(cat_ftr_col)
        
        output_tf_list.append(cat_ftr)
        
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

import functools

def create_tf_numeric_feature(col, MEAN, STD, default_value=0.0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
 
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,
                                                          default_value=default_value,
                                                          normalizer_fn=normalizer,
                                                          dtype=tf.dtypes.float32)
    
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    df['pred_binary'] = (df['pred_mean'] >= 5).astype(int)
    student_binary_prediction = df['pred_binary'].values
    
    return student_binary_prediction
