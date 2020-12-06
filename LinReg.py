import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

#    I could not get the index_col and usecols quite right at the beginning, and it prevented me from getting to dfNext
#    which drops all of the variables not needed for the regression.
#    Once the index_col is right and the other variables have been dropped, I think it should work.


# df = pd.read_csv(filepath_or_buffer='FinalCSV.csv',index_col=1, usecols=range(10), parse_dates=True, infer_datetime_format=True)

df = pd.read_csv('FinalCSV.csv') # for some reason when I read in the data this way it adds 7 columns of NAs after
drop_cols = np.arange(10,17) # the 'sentiment' column, so I drop these here
df.drop(df.columns[drop_cols], axis=1, inplace=True)
# print(df.head())
# Ticker,Company,Tweet,RoundDate,RoundTime,ID,Open,Close,PctChange,Sentiment

correlation = df.corr()
sb.heatmap(correlation)
# plt.show()

from project import WOE   #My directory is "Project" and I moved WOE.py from HW10 to run this
finalIV,IV = WOE.data_vars(df,df["PctChange"])
# IV.to_excel("IVOutput.xlsx")

dfNext = df.drop(['Company','Ticker','Tweet','RoundDate','RoundTime','ID','Open','Close'],axis=1)
# print(dfNext.head())
sentiment = pd.get_dummies(dfNext["Sentiment"])
dfNext.drop('Sentiment',axis=1,inplace=True)
dfReady = pd.concat([dfNext,sentiment],axis=1)

model1 = smf.ols(formula='PctChange ~ Negative', data=dfReady)
results1 = model1.fit()
print(results1.summary())

model2 = smf.ols(formula='PctChange ~ Positive', data=dfReady)
results2 = model2.fit()
print(results2.summary())