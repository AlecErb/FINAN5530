import numpy as np
import pandas as pd


# Import & Clean Trump Tweets
df = pd.read_csv(filepath_or_buffer='/Users/alec/Desktop/PythonforFinanceCode/FinalProject/trumpTweets.csv', index_col='id', parse_dates=True, infer_datetime_format=True)


startDate = '2017-01-21'
endDate = '2020-11-01'

df = df.drop(['device'], axis=1)
df.sort_values(by='date', inplace=True, ascending=False) 
df.text = df.text.str.lower()
df = df[ df['date'] > '2017-01-21 00:00:00'] # Only considering from inaguartion date (January 21, 2017) to November 6, 2020 (The day this data was colelcted)
df.drop(['isRetweet', 'isDeleted'], axis=1, inplace=True)
# df = df[ df['favorites'] > 300000 ]

# Import Stock Data
dfSP = pd.read_csv(filepath_or_buffer='/Users/alec/Desktop/PythonforFinanceCode/FinalProject/S&P500.csv')
dfSP.Security = dfSP.Security.str.lower()
securityArray = dfSP.Security.tolist()




# Check if tweet contains the name of an S&P company
npText = df['text'].to_numpy()
npSecurity = np.array(securityArray)
results = []
tweetedsecurity = []
count=0
# for i in range(len(npSecurity)):
#     results.append(list(filter(lambda x: npSecurity[i] in x, npText)))

for i in range(len(npSecurity)): # Removed Security's such as 'Visa', 'Ball', 'PPL', and 'GAP' because they can be commonly mistaken for their non-company counterparts
    for j in range(len(df)):
        if(npSecurity[i] in npText[j]):
            results.append(npText[j])
            count+=1
            tweetedsecurity.append(npSecurity[i])

print(results, count, tweetedsecurity)