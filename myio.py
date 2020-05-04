import csv
import numpy as np
from datetime import datetime
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from losses import mse, mse_prime
from charts import plotResult


def readData():
    count = 0
    data = []
    days = []
    with open('../../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22-complete-refined.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if count > 0:
                column = []
                days.append(row[0])
                for col in range(1, len(row)):
                    column.append(float(row[col]))
                data.append(column)
            count += 1

    return data, np.array(days)

def makeTrainingBatch(data, days):
    trainingBatch = []
    for i in range(len(data) - days):
        temp = []
        for j in range(i, i + days):
            x = 0
            for col in data[j]:
                temp.append(col)
                x += 1
        trainingBatch.append(temp)

    return np.array(trainingBatch)

def normalise(data):
    array = np.array(data)
    normalised = []
    mins = []
    maxs = []
    minmaxs = []

    for i in range(len(data[0])):
        mins.append(np.min(array[:,i]))
        maxs.append(np.max(array[:,i]))
        minmaxs.append(maxs[i] - mins[i])
    
    for i in range(len(data)):
        temp = []
        for j in range(len(data[i])):
            x = (data[i][j] - mins[j]) / minmaxs[j]
            temp.append(x)
        normalised.append(temp)

    return (np.array(normalised), mins, minmaxs)

def toDate(ts):
    dates = []
    for t in ts:
        dates.append(datetime.utcfromtimestamp(t * 86400).strftime('%Y-%m-%d'))
    return dates

def makeBatch(data, intervals):
    size = int(len(data) / intervals)
    x = []

    for i in range(intervals):
        x.append(data[i*size:(i+1)*size])

    return np.array(x)

def getSelection(arr, ex):
    selection = []
    for i in range(len(arr)):
        if (i != ex):
            if (selection == []):
                selection = arr[i]
            else:
                selection = np.concatenate((selection, arr[i]), 0)
    return selection

def crossValidation(trainingData, atr, days, DAYS_AHEAD, intervals=10, EPOCHS=300, LEARNING_RATE=0.2):
    reshaped_x = np.reshape(trainingData, (len(trainingData), 1, atr * days))
    reshaped_y = np.reshape(trainingData[:,2], (len(trainingData), 1, 1))

    x = makeBatch(reshaped_x, intervals)
    y = makeBatch(reshaped_y, intervals)

    errors = []
    for i in range(len(x)):
        x_train = getSelection(x, i)[:-DAYS_AHEAD]
        y_train = getSelection(y, i)[DAYS_AHEAD:]

        x_test = x[i][:-DAYS_AHEAD]
        y_test = y[i][DAYS_AHEAD:]

        inputSize = 2 * days
        outputSize = int(inputSize / 2)
        net = Network()
        net.add(FCLayer(inputSize, outputSize * 3))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(outputSize * 3, outputSize))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(outputSize, 1))
        net.add(ActivationLayer(tanh, tanh_prime))

        # train
        net.use(mse, mse_prime)
        net.fit(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)

        # test
        out, err = net.predict(x_test, y_test)
        errors.append(err)
        print(err)
    print(np.average(errors))
