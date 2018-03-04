# -*- coding: utf-8 -*-

# 誤差関数（l1正則化最小二乗法）による回帰分析
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal
from numpy import dot
 
#パラメータ
N = 50      # サンプルを取得する位置 x の個数
h = 0.2
lamda = 0.01
# データセット {x_n,y_n} (n=1...N) を用意
def create_dataset(num):
    dataset = DataFrame(columns=['x','y'])    #カラムがx,yのデータフレームをつくる
    for i in range(num):
        x = float(i)/float(num-1)     #1をnumで分割してxにする
        y = np.sin(2*np.pi*x) + normal(scale=0.3)      #sin2πxに正規分布の標準偏差0.3の乱数を与える
        dataset = dataset.append(Series([x,y], index=['x','y']),         #dataframeにSeriesをappendする
                                 ignore_index=True)
    return dataset
 
dataset = create_dataset(N)
 
#カーネル関数
def K(x,y):
    return np.exp(- (x - y) ** 2 / (2 * h ** 2))
 
#カーネル行列をもとめる
kernel = DataFrame()
for i in range(N):
    kernel = pd.concat([kernel, K(dataset.x[i], dataset.x)], axis = 1)
kernel.columns = range(N)   #カーネル行列完成
tmp = np.linalg.inv(kernel)   #カーネル行列の逆行列

# パラメータの更新  
theta = z = u = np.zeros(N)   #N個の要素の0ベクトルをつくる
paramhist = DataFrame()  #paramhistという空のdfつくって更新ごとにappendしていく
# iterationを10とする
for i in range(10):
    theta = dot(np.linalg.inv(dot(kernel, kernel) + np.eye(N)), dot(kernel, dataset.y) + z - u)   #θの更新
    arr1 = theta + u - lamda * np.ones(N)
    arr2 =  - theta - u - lamda * np.ones(N)
    z = np.array(list(map(lambda x: max(x,0), arr1))) - np.array(list(map(lambda x: max(x,0), arr2)))
    u = u + theta - z
    paramhist = paramhist.append(   #更新し終えたらappendする(これを30回appendする)
                    Series(theta),
                    ignore_index=True)

# 図の表示
def f(x):    #モデル関数
    y = 0
    for i, w in enumerate(theta):        #インデックス付き処理をする
        y += w * K(x, dataset.x[i])
    return y
 
fig = plt.figure()
plt.xlim(-0.05,1.05)
plt.ylim(-1.5,1.5)
 
# データセットを表示
plt.scatter(dataset.x, dataset.y, marker='o', color='blue')      #点が表示される

# 真の曲線を表示
linex = np.arange(0,1.01,0.01)     #配列をつくる
liney = np.sin(2*np.pi*linex)
plt.plot(linex, liney, color='green', linestyle='--')

# 近似の曲線を表示
linex = np.arange(0,1.01,0.01)
liney = f(linex)                                    #求めたfを使って曲線をかく
plt.plot(linex, liney, color='red')
 
fig.show()