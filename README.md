# EX-no-5-Implement-Auto-Regression-Model-in-Python
## AIM:
To Implement Auto Regression Model in Python
## STEPS:
## PROGRAM:
```
!pip install statsmodels --upgrade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
df = pd.read_csv('/content/rainfall.csv', index_col=0, parse_dates=True)
df.shape
df.pop("preciptype")
df.pop("dew")
df.pop("humidity")
df.pop("sealevelpressure")
df.pop("winddir")
df.pop("solarradiation")
df.pop("windspeed")
df.pop("precipprob")
df.head()
x=df.values
df.plot()
from statsmodels.tsa.stattools import adfuller
dftest= adfuller(df['temp'],autolag='AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ",dftest[1])
print("3. Number Of Lags : ",dftest[2])
print("4.Num of observation used FOr ADF Regression  and Critical value Calculation :",dftest[3])
for key,val in dftest[4].items():
     print("\t",key, ":",val)
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
pacf=plot_pacf(df['temp'],lags=25)
acf=plot_acf(df['temp'],lags=25)
train=x[:len(x)-600]
test=x[len(x)-600:]
model=AutoReg(train,lags=40).fit()
print(model.summary())
pred=model.predict(start=len(train),end=len(x)-1,dynamic=False)
from matplotlib import pyplot
pyplot.plot(pred)
pyplot.plot(test,color='green')
print(pred)
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=sqrt(mean_squared_error(test,pred))
rmse
pred_future=model.predict(start=len(x)+1,end=len(x)+7,dynamic=False)
print("future prediction")
print(pred_future)
```
## OUTPUT:

### 1.df.shape

![shape](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/e87098d4-4eab-4794-a1c7-dc40f1fbde1e)
### 2.df.head()

![head](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/924990aa-5033-4d93-9fac-7cf1bc2b5959)

### 3.df.plot()

![3](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/1e7abcae-fc95-4181-b569-d03dbb056cde)



### 4.adfuller


![4](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/46cce841-ecf3-4753-976c-fc436ed58457)


### 5.Partial Autocorrelation

![5](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/576f456e-5a2d-4f9a-a3cb-69f4edb7f1cd)


### 6.Autocorrelation
![6](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/343497d1-1b78-45de-b80c-5e78196c4c07)

### 7.Autoreg Model Result

![7](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/a0ef3e82-97cd-4db3-83a8-451012c2454b)

### 8.Predicted Values and Plot
![8](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/67ca56f8-b401-4f34-8722-ea4d3dd1ab04)

![9](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/176f0b0d-ec16-45e0-a0f5-284edd10a93c)

### RMSE Value

![10](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/ebe5ffa9-ad45-4bfb-8c74-58e91cabc0dc)
### Future Prediction
![11](https://github.com/praveenst13/EX-no-5-Implement-Auto-Regression-Model-in-Python/assets/118787793/6e1cb07c-0721-4be2-a3cf-ad468d27a796)



## RESULT:
