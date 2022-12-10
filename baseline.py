
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from sklearn import svm
import math
import sys
import pandas_datareader.data as web
from sklearn.metrics import r2_score,explained_variance_score,mean_squared_error
#%%
# start = datetime(2017, 1, 1)
# end = datetime(2022, 12, 1)
# symbols = ['^NDQ','10CNY.B', 'TSLA.US']
# for i in symbols:
#     web.DataReader(i, 'stooq', start=start, end=end).to_csv('data/'i+'_2017_2022.csv')
#%%

ndq = pd.read_csv('./^NDQ_2017_2022.csv').set_index('Date')
cny = pd.read_csv('./10CNY.B_2017_2022.csv').set_index('Date')
tsla = pd.read_csv('./TSLA.US_2017_2022.csv').set_index('Date')
dataset = [ndq, cny, tsla]
percent = [0.5,0.6,0.7,0.9]
arch = []
garch = []
for per in percent:
    print(f'current split is {per}')
    for i in range(1):
        #%%
        # Calculate first order difference of log of price
        # Rescale by 100 for fitting the model
        returns = 100 * dataset[i].Close.pct_change().dropna()
        returns = returns[::-1]
        vio = returns**2
        # plot_pacf(vio)
        # plt.savefig(f'./pacf_dataset_{i}')
        # plt.show()

        # train, test = ndq[:int(0.6*len(vio))], ndq[int(0.6*len(vio)):]
        ##########typical arch model###############
        point = int(per*len(returns))
        am = arch_model(returns, mean='AR',lags=8,vol='ARCH',p=8)
        y = vio[point:]
        res = am.fit(last_obs=point+1, disp="off")
        # for i in range(point):
        #     temp = res.forecast(start=i, horizon=1, reindex=False).variance.iloc[0]
        #     for_in.append(temp)
        # print(f'in_sample result is {explained_variance_score(y,for_in)}')
        var = res.forecast(reindex=False).variance
        s = format(r2_score(y,var), '.5f')
        # print(res)
        print(f'ARCH real result of set{i} is {s}')
        arch.append(float(s))

        # print(pd.DataFrame(forecasts).T)
        ##########GARCH model################

        am = arch_model(returns, vol="GARCH")
        y = vio[point:]
        res = am.fit(last_obs=point+1, disp="off")
        # for i in range(point):
        #     temp = res.forecast(start=i, horizon=1, reindex=False).variance.iloc[0]
        #     for_in.append(temp)
        # print(f'in_sample result is {explained_variance_score(y,for_in)}')
        # for i in range(point,len(returns)):
        #     temp = res.forecast(start=i, horizon=1,reindex=False).variance.iloc[0]
        #     forecasts.append(temp)
        var = res.forecast(reindex=False,start=point).variance
        s = format(r2_score(y,var),'.5f')
        # print(res)
        print(f'GARCH real result of set{i} is {s}')
        garch.append(float(s))



import matplotlib.pyplot as plt
import numpy as np
import random
# 设置显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
# plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

#设置画布大小像素点
# plt.figure(figsize=(10,1),dpi=100)

all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=10)

choice = 0
figure = [["ARCH","R-SQUARE"],["GARCH","R-SQUARE"],["SVM-RBF","R-SQUARE"],["SVM-LINEAR","R-SQUARE"]]

Xline = ["50/50", "60/40", "70/30", "90/10"]

Yline = [arch,
        garch,
        [0.83986, 0.46419, 0.41995, 0.27316],
        [-0.00548, 0.40266, 0.43924, 0.42935]]

plt.figure(figsize=(10,8),dpi=100)

# fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
# ax.hlines(y=Yline[1], xmin=11, xmax=26, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
# ax.scatter(y=Yline[1], x=Xline, s=75, color='firebrick', alpha=0.7)
# plt.show()

for choice in range(4):
    plt.subplot(2,2,choice+1)
    plt.plot(Yline[choice],Xline,lw=3.0,ls="--",marker='o',mfc='orange',color = 'firebrick')
    plt.title("", fontsize=11)
    plt.ylabel(figure[choice][0],fontsize=12)
    plt.xlabel(figure[choice][1],fontsize=12)
    plt.xlim(left = min(Yline[choice])-0.01,right = max(Yline[choice])+0.003)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.5, hspace=0.5)
plt.savefig('./split_ablation')
plt.show()

# print(pd.DataFrame(forecasts).T)

