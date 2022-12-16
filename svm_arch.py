import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import r2_score,explained_variance_score,mean_squared_error
import pickle
import math
import sys
#

def construct_window_n(df,percent):
    X = []
    y = []
    vio = (df ** 2)
    sigma_t = []
    for i in range(5, len(vio)):
        sigma_t.append((np.mean(vio[i - 5:i])))
    sigma_t = np.sqrt(sigma_t)
    for i in range(5,len(sigma_t)):
        a = []
        for j in range(1,5):
            a.append(sigma_t[i-j])
        # a.append(i)
        X.append(a)
        y.append(sigma_t[i])
    train = X[:int(percent*len(X))] , y[:int(percent*len(X))]
    test = X[int(percent*len(X)):] , y[int(percent*len(X)):]
    return train,test

#baseline_svm
# def construct_window_n(df,percent):
#     X = []
#     y = []
#     #print(df)
#     vio = (df ** 2)
#     #print(vio)
#     sigma_t = []
#     for i in range(5, len(vio)):
#         sigma_t.append((np.mean(vio[i - 5:i])))
#     #print(sigma_t)
#     #sigma_t = np.square(sigma_t)
#     #print(sigma_t)
#     for i in range(0,len(sigma_t)-1):
#         a = []
#         a = [np.sqrt(sigma_t[i]), df[i+4]]
#         # a.append(i)
#         X.append(a)
#         y.append(sigma_t[i+1])
#     train = X[:int(percent*len(X))] , y[:int(percent*len(X))]
#     test = X[int(percent*len(X)):] , y[int(percent*len(X)):]
#     return train,test

ndq = pd.read_csv('./data/^NDQ_2017_2022.csv').set_index('Date')
cny = pd.read_csv('./data/10CNY.B_2017_2022.csv').set_index('Date')
tsla = pd.read_csv('./data/TSLA.US_2017_2022.csv').set_index('Date')
# Calculate first order difference of log of price
# Rescale by 100 for fitting the model

dataset = [ndq, cny, tsla]
percent = [0.5,0.6,0.7,0.9]
# C = [x for x in range(5,50,5)]
linear = []
rbf = []
for i in range(3):

        returns =  100*dataset[i].Close.pct_change().dropna()
        returns = returns[::-1]
        trainset ,testset = construct_window_n(returns,0.6)

        svr_rbf = SVR(kernel='rbf',C=40)
        model1 = svr_rbf.fit(trainset[0],trainset[1])
        # f = open(f'rbf_dataset{i}_window{n}.pickle', 'wb')
        # # pickle.dump(model1,f)
        # # f.close()
        y_rbf = model1.predict(testset[0])
        rbf_score = format(r2_score(testset[1],y_rbf),'.5f')
        print(f'rbf_score of set {i} is {rbf_score}')
        rbf.append(float(rbf_score))
        # c = [x for x in range(100,300,10)]
        res = []

        svr_sigmoid = SVR(kernel='linear',C=5)
        model2  = svr_sigmoid.fit(trainset[0],trainset[1])

        y_sig = model2.predict(testset[0])
        sig_score = r2_score(testset[1],y_sig )
        sig_score = format(sig_score,'.5f')
        print(f'sig_score of set {i} is {sig_score}')
        linear.append(float(sig_score))
        #
        # svr_poly = SVR(kernel = 'poly',C=10)
        # model3 = svr_poly.fit(trainset[0],trainset[1])
        # # f = open(f'poly_dataset{i}_window{n}.pickle', 'wb')
        # # pickle.dump(model3, f)
        # # f.close()
        # y_poly = model3.predict(testset[0])
        # poly_score = format(r2_score(testset[1],y_poly),'.5f')
        # # print(f'poly_score_in of set {i} is {poly_score_in}')
        # print(f'poly_score of set {i} is {poly_score}')
        print(f'########## end of set {i}')


        x = range(len(y_rbf))
        # print(linear)
        # print(rbf)
        plt.style.use('ggplot')
        plt.subplot(1,2,1)
        plt.plot(x, testset[1], color='r', label='real_vol')
        plt.plot(x, y_rbf, lw=.5,color='b', label='svr_rbf')
        plt.title('SVR-RBF')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(x, testset[1], color='r', label='real_vol')
        plt.plot(x, y_sig,lw=.5,color='b', label='svr_linear')
        plt.title('SVR-LINEAR')
        plt.legend()
        plt.savefig(f'dataset{i}')
        plt.show()
# #
# linear = [0.46877, 0.46419, 0.44844, 0.43002, 0.42052, 0.40307, 0.38965, 0.38741, 0.38464]
# rbf = [0.09425, 0.2211, 0.31746, 0.40266, 0.43944, 0.48521, 0.50606, 0.52556, 0.52771]
# plt.subplot(1,2,1)
# plt.plot(C, linear,lw=3.0,ls="--",marker='o',mfc='orange',color = 'firebrick')
# plt.title("Effect of C in SVM-LINEAR", fontsize=11)
# plt.ylabel('R-SQUARE')
# plt.xlabel('Selection of C')
# plt.subplot(1,2,2)
# plt.plot(C, rbf, lw=3.0, ls="--", marker='o', mfc='orange', color='firebrick')
# plt.title("Effect of C in SVM-RBF", fontsize=11)
# plt.ylabel('R-SQUARE')
# plt.xlabel('Selection of C')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.5, hspace=0.5)
# plt.savefig('C effect')
# plt.show()
#
#
#
