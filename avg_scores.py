import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


__author__ = 'matteo'


def plot_red(year):

    y2005 = [29.1, 31.3, 32.1, 32.2, 35.1]
    y2006 = [32.2, 34.1, 34.4, 34.9, 36.8]
    if year==2005:
        y = y2005
    else:
        y = y2006

    x = ('LEAD','nGRAM','RNN','CNN','HUMAN')
    labels = np.arange(len(y))
    fig, ax = plt.subplots()

    ax.set_ylabel('% Rouge F-Score')
    ax.set_xlabel('Algorithm')

    if year==2005:
        ax.set_title('System performance with different redundancy metrics, 2005')
    else:
        ax.set_title('System performance with different redundancy metrics, 2006')

    ax.set_xticks(labels+0.8/2)
    ax.set_xticklabels(x)

    if year==2005:
        ax.set_ylim([27,35.5])
    else:
        ax.set_ylim([30,37.3])
    barlist = plt.bar(labels, y)

    barlist[0].set_color('b')
    barlist[1].set_color('g')
    barlist[2].set_color('r')
    barlist[3].set_color('r')

    plt.show()


def plot_NL_detail():

    fig, ax = plt.subplots()

    ax.set_ylabel('% Rouge F-Score')
    ax.set_xlabel('Test Collections')
    ax.set_title('Systems performance on specific collections')

    x = np.arange(5)
    LIN = [29.0, 40.2, 28.5, 26.5, 26.4]
    DT  = [30.6, 39.9, 28.7, 26.9, 27.0]
    KRR = [28.1, 41.5, 29.2, 25.0, 27.0]
    RF  = [33.0, 45.1, 30.1, 27.4, 30.6]
    GBR = [33.6, 41.8, 29.0, 25.7, 28.6]

    plt.plot(x,LIN,label="Lin Reg")
    plt.plot(x,DT, label="Decision Trees")
    plt.plot(x,KRR, label="Support Vector Regression")
    plt.plot(x,RF, label="Random Forests")
    plt.plot(x,GBR, label="Gradient Boosting")
    plt.legend()

    ax.set_ylim([24,47])

    plt.show()


def plot_NL():

    y = [29.1, 30.0, 30.5, 31.3, 31.2, 30.5]
    x = ('LEAD','LR','DT','RF','GB','SVR')
    labels = np.arange(6)
    fig, ax = plt.subplots()

    ax.set_ylabel('% Rouge F-Score')
    ax.set_xlabel('Algorithm')
    ax.set_title('System performance with different regression algorithms')

    ax.set_xticks(labels+0.8/2)
    ax.set_xticklabels(x)
    ax.set_ylim([26,32.1])
    barlist = plt.bar(labels, y)

    barlist[0].set_color('b')
    barlist[1].set_color('g')
    barlist[2].set_color('r')
    barlist[3].set_color('r')
    barlist[4].set_color('r')
    barlist[5].set_color('r')

    plt.show()


def plot_allFeat(year):

    if year == 2005:
        y = [29.1, 31.5, 31.7, 35.2]
    else:
        y = [32.2, 34.4, 34.1, 36.9]

    x = ('LEAD','ALL(noTFIDF)','ALL(+TFIDF)','HUMAN')
    labels = np.arange(4)
    fig, ax = plt.subplots()

    ax.set_ylabel('% Rouge F-Score')
    ax.set_xlabel('Algorithm')

    if year==2005:
        ax.set_title('System performance with the additional features, 2005')
    else:
        ax.set_title('System performance with the additional features, 2006')

    ax.set_xticks(labels+0.8/2)
    ax.set_xticklabels(x)

    if year==2005:
        ax.set_ylim([28,36])
    else:
        ax.set_ylim([31,38])

    barlist = plt.bar(labels, y)
    barlist[0].set_color('b')
    barlist[1].set_color('r')
    barlist[2].set_color('r')

    plt.show()


plot_allFeat(2005)
plot_allFeat(2006)
exit()
plot_red(2005)
plot_red(2006)
plot_NL()
plot_NL_detail()