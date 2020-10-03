# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:29:55 2020

@author: Chetan
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf 
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots

data = pd.read_excel("D:\\chetan\\assignment\\17.forcasting/Airlines+Data.Xlsx")
data.shape
data.columns
data.info()
data.describe()
plt.plot('Month','Passengers',data = data)
#from plot it is indentified that it is multiplicative seasonality
data['t']=np.arange(1,97,1)
data["log_p"]=np.log(data['Passengers'])

sns.boxplot(data.Month,data.Passengers)
data.index = pd.to_datetime(data.Month,format="%b-%y")
data["Date"] = pd.to_datetime(data.Month,format="%b-%y")
data["month"] = data.Date.dt.strftime("%b") 
data["year"] = data.Date.dt.strftime("%Y") 
data.shape

sns.boxplot(x="month",y="Passengers",data=data)
sns.boxplot(x="year",y="Passengers",data=data)
Train =data.head(80)
Test = data.tail(16)

################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('log_p~month',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea#135.326

##################Multiplicative Additive Seasonality ###########
Mul_Add_sea = smf.ols('log_p~t+month',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea#9.469 

def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

tsa_plots.plot_acf(data.Passengers)
tsa_plots.plot_pacf(data.Passengers)

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=8).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers) # 21.6
Test.info()

#simple exponential
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers) # 17.438

result =pd.DataFrame({"model":[" Multiplicative Seasonality","Multiplicative Additive Seasonality","Holts",'simple exponential'],
          'accuracy':[rmse_Mult_sea,rmse_Mult_add_sea,MAPE(pred_hwe_mul_add,Test.Passengers),MAPE(pred_ses,Test.Passengers)]})

result
#finally we are going to use Multiplicative Additive Seasonality