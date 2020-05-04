import numpy as np
from datetime import datetime

from myio import readData, normalise, makeTrainingBatch, toDate, crossValidation
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from losses import mse, mse_prime
from charts import plotResult

SIZE = 500 # testing data size
DAYS = 5 # days used for calculation
DAYS_AHEAD = 3 # day prognosis ahead
EPOCHS = 1000 # max epochs
LEARNING_RATE = 0.2 # learning rate

data, days = readData()
dataArray = np.array(data)[:,:-1]
normalised, mins, maxs = normalise(dataArray)
trainingData = makeTrainingBatch(normalised, DAYS)

ATR = len(dataArray[0]) # number of atributes
SIZE = len(trainingData) - 200

# crossValidation(trainingData, ATR, DAYS, DAYS_AHEAD, EPOCHS=EPOCHS)

# training data
x_train = np.reshape(trainingData, (len(trainingData), 1, ATR * DAYS))[0:SIZE - DAYS_AHEAD]
y_train = np.reshape(trainingData[:,2], (len(trainingData), 1, 1))[DAYS_AHEAD:SIZE]
time_train = dataArray[DAYS_AHEAD:SIZE]
x_test = np.reshape(trainingData, (len(trainingData), 1, ATR * DAYS))[SIZE:-DAYS_AHEAD]
y_test = np.reshape(trainingData[:,2], (len(trainingData), 1, 1))[SIZE + DAYS_AHEAD:]
time_test = dataArray[SIZE+DAYS+DAYS_AHEAD:]
inputSize = len(x_train)

# network
inputSize = ATR * DAYS
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
print(err)
outTrain, err = net.predict(x_train, y_train)

printingRes = np.concatenate(out, axis=0)
printingTrain = np.concatenate(outTrain, axis=0)

date_test = days[SIZE+DAYS+DAYS_AHEAD:]
date_train = days[DAYS_AHEAD:SIZE]

# plotResult(time_test[:,0], np.reshape(printingRes, len(printingRes)),  np.reshape(y_test, len(y_test)))
plotResult(date_test, np.reshape(printingRes, len(printingRes)) * maxs[1] + mins[1], time_test[:,0])
plotResult(date_train, np.reshape(printingTrain, len(printingTrain)) * maxs[1] + mins[1], time_train[:,0])