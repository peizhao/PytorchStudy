import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import logging as logger
logger.basicConfig(level=logger.INFO)

data_prefix = '../../../data/rnn/da_rnn'
data_file = 'nasdaq100_padding.csv'

def get_dataPath(prefix, fileName):
    return "{}/{}".format(data_prefix, data_file)

def getCSVDataValues(fileName, label):
    """
    Get the CSV file as value and labels
    :param fileName:  the csv file name
    :param label: the column name for the label
    :return: ValueMatrix, label Vector
    """
    dat = pd.read_csv(fileName)
    X = dat.loc[:, [x for x in dat.columns.tolist() if x != label]].as_matrix()
    Label = dat.loc[:, [x for x in dat.columns.tolist() if x == label]].as_matrix()
    return X, Label

def getTrainSize(values, rate):
    """
    Get the train size based on the r
    :param values: The Train value list
    :param ration: The train rate on all the data
    :return: training dat size
    """
    return values.shape[0]*rate

if __name__ == "__main__":
    X, Y = getCSVDataValues(get_dataPath(data_prefix,data_file),'NDX')
    train_size = getTrainSize(X,0.7)
    print(X.shape)
    print(Y.shape)
    print(train_size)
