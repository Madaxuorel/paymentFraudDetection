import pandas as pd
import numpy as np
from tqdm import tqdm

def loadData(name):
    data = pd.read_csv(name)
    data.drop("step",axis=1,inplace=True)
    data.drop("isFlaggedFraud",axis=1,inplace=True)
    return data

"""
qualitative to quant data
"""
def quantitate(data,pastFraud):
    thisData = data.copy()
    """1 : change type to 
    1 = payment, 
    2 = cash out, 
    3 = transfer, 
    4 = cash in, 
    5 = debit"""

    thisData['transactionTypeCode'] = thisData['type'].map(lambda x: 
    1 if x == 'PAYMENT' else
    2 if x == 'CASH_OUT' else
    3 if x == 'TRANSFER' else
    4 if x == 'CASH_IN' else
    5 if x == 'DEBIT' else
    None)
    thisData.drop("type",axis=1,inplace=True)

    """
    implement check if origin or destination adress involved in past fraud
    """
    thisData["originPastFraud"] = thisData["nameOrig"].isin(pastFraud).astype(int)

    thisData["destinationPastFraud"] = thisData["nameDest"].isin(pastFraud).astype(int) 
    
    thisData.drop("nameOrig",axis=1,inplace=True)

    thisData.drop("nameDest",axis=1,inplace=True)

    print(thisData)

    return thisData


"""
get list of all addresses involved in fraud operations in the train dataset
"""
def flagedAdresses(transactionData, isFraud):

    flagedAdresses = set()

    transactionData["isFraud"] = isFraud
    for index, row in tqdm(transactionData.iterrows()):
        if row['isFraud'] == 1:
            flagedAdresses.add(row["nameOrig"])
            flagedAdresses.add(row["nameDest"])

    transactionData.drop("isFraud",axis=1,inplace=True)
    print(transactionData)
    return flagedAdresses

    

"""
split into 2 sets, with X and Y split
"""
def historySplit(data, ratio):
    numberOfRows = data.shape[0]
    numberOfRowsTest = int(np.floor(numberOfRows * ratio))
    numberOfRowsTrain = numberOfRows - numberOfRowsTest

    testData = data.iloc[:numberOfRowsTest, :]
    trainData = data.iloc[numberOfRowsTest:numberOfRows, :]

    X_train = trainData.iloc[:, :-1]
    Y_train = trainData.iloc[:, -1]

    X_test = testData.iloc[:, :-1]
    Y_test = testData.iloc[:, -1]

    return X_train, Y_train, X_test, Y_test
    