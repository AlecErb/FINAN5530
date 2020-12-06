import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

#    I could not get the index_col and usecols quite right at the beginning, and it prevented me from getting to dfNext
#    which drops all of the variables not needed for the regression.
#    Once the index_col is right and the other variables have been dropped, I think it should work.


# df = pd.read_csv(filepath_or_buffer='FinalCSV.csv',index_col=1, usecols=range(10), parse_dates=True, infer_datetime_format=True)
print(df.head())
# Ticker,Company,Tweet,RoundDate,RoundTime,ID,Open,Close,PctChange,Sentiment

correaltion = df.corr()
sb.heatmap(correaltion)
# plt.show()

from Project import WOE   #My directory is "Project" and I moved WOE.py from HW10 to run this
finalIV,IV = WOE.data_vars(df,df["PctChange"])
# IV.to_excel("IVOutput.xlsx")

dfNext = df.drop(['Company','Ticker','Tweet','RoundDate','RoundTime','ID','Open','Close'],axis=1)

sentiment = pd.get_dummies(dfNext["Sentiment"])
dfNext.drop('Sentiment',axis=1,inplace=True)
dfReady = pd.concat([dfNext,sentiment],axis=1)
print(dfReady.head())


dfResults = dfReady["PctChange"]
dfInputs = dfReady.drop("PctChange",axis=1)
print(dfResults.head())
print(dfInputs.head())

inputsTrain,inputsTest,resultTrain,resultTest = train_test_split(dfInputs,dfResults,test_size=0.3,random_state=1)

LogReg = LogisticRegression()
LogReg.fit(inputsTrain,resultTrain)
print("Coefs(Mns):",LogReg.coef_)
print("Intercept(b):",LogReg.intercept_)
resultPred = LogReg.predict(inputsTest)

print(confusion_matrix(resultTest,resultPred))
print(classification_report(resultTest,resultPred))
