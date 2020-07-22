import os
from Utils.csvread import CsvReader
import pandas as pd
import numpy as np

csvreader = CsvReader()
path = os.getcwd()
path = path + '/Results/'
folders = os.listdir(path)

data = None
result = None
#find all the results files
for folder in folders:
    files = os.listdir(path+folder)
    files.sort()
    for file in files:
        if '.csv' in file:

            #go through all the results
            if 'results' in file:
                tmpresult = csvreader.read(path+folder+'/'+file)
                file = file[:-4]
                split_list = file.split(sep='_')
                tmpresult['model'] = split_list[2]
                tmpresult['symbol'] = split_list[1]
                if result is None:
                    result = tmpresult
                else:
                    result = pd.concat([result, tmpresult])
            #go through all the predictions
            else:
                tmpdata = csvreader.read(path+folder+'/'+file)
                file = file[:-4]
                split_list = file.split(sep='_')
                tmpdata['model'] = split_list[1]
                tmpdata['symbol'] = split_list[0]
                if data is None:
                    data = tmpdata
                else:
                    data = pd.concat([data, tmpdata])

prediction_save_path = path + 'merged_predictions.csv'
result_save_path = path + 'merged_results.csv'
data.to_csv(prediction_save_path, index = False)
result.to_csv(result_save_path, index = False)
