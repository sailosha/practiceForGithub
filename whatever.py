import tensorflow as tf
import csv
import pandas as pd
import os
import shutil
import numpy as np
'''
creat by roy
i changed it on 2019-10-15 16:00:27

'''

file_name_string="1p6ft.csv"
CSV_COLUMNS=[]

'''
reading classmethod
'''
# with open('D:/Data2019/data combination/11111.csv', 'r') as dataset:
#     reader = csv.reader(dataset)
#     for row in reader:
#        d['label' ] = row[0:]
# print(d)

'''
read first row of data
'''
with open('D:/Data2019/data combination/11111.csv', 'r') as dataset:
    reader = csv.reader(dataset)
    CSV_COLUMNS = next(reader)

# print (CSV_COLUMNS)

'''
copy from google
thanks google and coursera to offer this lab
'''
#label : result to predict
#freature : data input
LABEL = CSV_COLUMNS[64]

FEATURES = CSV_COLUMNS[0:len(CSV_COLUMNS)-1]
label_keys = ['Absence', 'nothing']

df_train = pd.read_csv('D:/Data2019/data combination/train80.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('D:/Data2019/data combination/valid20.csv', header = None, names = CSV_COLUMNS)


def make_train_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = df[LABEL],
        batch_size = 128,
        num_epochs = num_epochs,
        shuffle = True,
        queue_capacity = 1000
    )

def make_prediction_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = df[LABEL],
        batch_size = 128,
        shuffle = False,
        queue_capacity = 1000
    )

def make_feature_cols():
    input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    return input_columns


tf.logging.set_verbosity(tf.logging.INFO)
OUTDIR = 'D:/Data2019/data combination/taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.DNNClassifier(
    feature_columns=make_feature_cols(),
    hidden_units = [30, 18, 6],
    n_classes=2,
    label_vocabulary=label_keys,
    # optimizer= tf.train.AdamOptimizer(
    optimizer = tf.train.FtrlOptimizer(0.01, l1_regularization_strength=0.01, l2_regularization_strength=0.01),
    model_dir = OUTDIR)


model.train(input_fn = make_train_input_fn(df_train, num_epochs = 100),steps=3300)



def print_rmse(model, df):
    metrics = model.evaluate(input_fn = make_prediction_input_fn(df))
    print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))


print_rmse(model, df_valid)
