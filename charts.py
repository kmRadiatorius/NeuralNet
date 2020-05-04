import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from mpl_toolkits import mplot3d

def plotResult(x, y1, y2):
    plt.figure(figsize=(30, 15))
    plt.plot(x, y1, label='Results')
    plt.plot(x, y2, label='Real Data')
    plt.title('Bitcoin Price')
    plt.xlabel('Time', )
    plt.ylabel('Price')
    plt.xticks(x[::7], fontsize=12, rotation='vertical')
    plt.yticks(fontsize=12)
    plt.legend()
    plt.show()