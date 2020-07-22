import os
from Utils.csvread import CsvReader
import pandas as pd
import numpy as np


csvreader = CsvReader()

#read the raw dataset
alldata = csvreader.read('example_data.csv')

#find all the identifiers
identifier_list = alldata.symbol.unique()

#split the whole dataset for 7:3
index_list = alldata.week.unique()
splitter = int(index_list[-1]*0.7)
tmp_training_data = alldata['week'] <= splitter
tmp_training_data = alldata[tmp_training_data]
tmp_test_data = alldata['week'] > splitter
tmp_test_data = alldata[tmp_test_data]

#go through all the identifiers
for identifier in identifier_list:
    #split the raw dataset into training and testing datasets
    test_data = tmp_test_data['symbol'] == identifier
    test_data = tmp_test_data[test_data]
    training_data = tmp_training_data

    #get file path
    path = os.getcwd()
    path = path + '/Results/' + identifier

    #create folder
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    resultpath = path + '/results_' + identifier
    predictionpath = path + '/' + identifier
    figurepath = path + '/' + identifier

    #create datasets
    test_data.to_csv(r'test_data.csv', index = False)
    training_data.to_csv(r'training_data.csv', index = False)

    #run ml_platform
    os.system('sudo python3 ml_platform.py -train training_data.csv -in input.txt -out output.txt -t r -kfold 5 -index indexarray.txt -test test_data.csv -rp ' + resultpath + \
    ' -pp ' + predictionpath + ' -fp ' + figurepath)

    #remove the datasets
    os.remove('test_data.csv')
    os.remove('training_data.csv')
