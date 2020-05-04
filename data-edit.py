import csv
import numpy as np
from datetime import datetime

def readData():
    count = 0
    data = []
    with open('../../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22-complete.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if count > 0:
                column = []
                for col in row:
                    column.append(float(col))
                data.append(column)
            count += 1
    return data

def writeData(data):
    with open('../../bitstampUSD_1-min_data_2012-01-01_to_2020-04-22-complete-refined.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in data:
            writer.writerow([row[0], row[1], row[2], row[3]])

data = readData()
newData = []
tempRows = []
for i in range(len(data) - 1):
    row = data[i]
    thisDay = datetime.utcfromtimestamp(row[0]).strftime('%Y-%m-%d')
    nextDay = datetime.utcfromtimestamp(data[i + 1][0]).strftime('%Y-%m-%d')
    if (thisDay == nextDay):
        tempRows.append(row)
    else:
        tempRows.append(row)
        arr = np.array(tempRows)
        tempData = []
        high = np.max(arr[:,2])
        low = np.min(arr[:,3])
        tempData.append(row[0] / 24 / 3600)
        tempData.append(thisDay)
        tempData.append(high)
        tempData.append(low)
        newData.append(tempData)
        tempRows = []
    day = thisDay

# print(len(newData))
writeData(newData)

