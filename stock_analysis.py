import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

start = datetime.datetime(2010, 1, 10)
#end = datetime.datetime.now()
end = datetime.datetime(2017, 1, 11)
df = web.DataReader("AAPL", 'yahoo', start, end)
#df.head()
df.tail()

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AMZN')
mavg.plot(label='mavg')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
#mpl.rc('figure', figsize=(8, 7))
mpl.rc('figure', figsize=(15, 9))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
plt.show()

# return deviation
rets = close_px / close_px.shift(1) - 1
rets.head()

rets.plot(label='return')
plt.show()

# analysing competitor stocks
dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
dfcomp.tail()

# correlation analysis
retscomp = dfcomp.pct_change()
corr = retscomp.corr()
corr.tail()

# apple and GE return distribution
plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel('Returns AAPL')
plt.ylabel('Returns GE')
plt.show()

# Kernel Density Estimate (KDE)
pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));
plt.show()

# heat map
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);
plt.show()

# stock returns rate and risk
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.show()

# feature engineering
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

dfreg.head()

import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
print(dfreg.shape)
#print(dfreg)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

print('Dimension of X',X.shape)
#print(X)
print('Dimension of y',y.shape)

# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# model generation
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

# evaluation
confidencereg = clfreg.score(X_test, y_test)
print('The linear regression confidence is {}'.format(confidencereg))

confidencepoly2 = clfpoly2.score(X_test,y_test)
print('The quadratic regression 2 confidence is {}'.format(confidencepoly2))

confidencepoly3 = clfpoly3.score(X_test,y_test)
print('The quadratic regression 3 confidence is {}'.format(confidencepoly3))

confidenceknn = clfknn.score(X_test, y_test)
print('The knn regression confidence is {}'.format(confidenceknn))

# top 3 models are
# 1. linear regression: 0.9715671136865407
# 2. quadratic regression 3: 0.9536641707905672
# 3. quadratic regression 2: 0.9493183299155221
# *Score numbers above are changing everytime you run this program. This is the numbers when I ran.

# stock forecast
forecast_set = {
    "linear regression": clfreg.predict(X_lately),
    "quadratic regression 3": clfpoly3.predict(X_lately),
    "quadratic regression 2": clfpoly2.predict(X_lately)
}
#print(forecast_set)

# get last_date
last_date = dfreg.iloc[-1].name

for k, v in forecast_set.items():
    dfreg['Forecast'] = np.nan
    #print(dfreg['Forecast'])
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)
    print(last_date)
    #for i in forecast_set:
    for i in v:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.title(k)
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
