# encoding=utf-8
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import cm as cm


def load_iris():
    df = pd.read_csv('iris.txt', sep=',', names=['sl', 'sw', 'pl', 'pw', 'label'])
    # print data[['sl', 'sw', 'pl', 'pw']]
    df = df.sort_values(by=['label'])
    # print df
    x = df[['sl', 'sw', 'pl', 'pw']]
    #x = preprocessing.scale(x)      # x[i,j] = (x[i,j] - avg(column_i)) / std(column_i)
    y = df[['label']]
    return x, y


def corr_matrix(x, y):
    corr = np.corrcoef(x)
    fig, ax = plt.subplots(figsize=(7, 7))
    cax = ax.matshow(corr)
    fig.colorbar(cax, ticks=[.4, .5, .6, .7, .8, .9])
    plt.show()
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)

    cax = ax1.imshow(np.corrcoef(x), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Iris Correlation Matrix')
    labels = ['Setosa', 'Versicolor', 'Virginica']
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    fig.colorbar(cax, ticks=[.4, .5, .6, .7, .8, .9])
    plt.show()
    """
    

if __name__ == "__main__":
    data, label = load_iris()
    corr_matrix(data, label)
