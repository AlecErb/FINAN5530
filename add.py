import numpy as np
import pandas as pd
import pandas_datareader.data as web
import spacy


# Import & Clean Trump Tweets
df = pd.read_csv(filepath_or_buffer='trumpTweets.csv', index_col='id', parse_dates=True, infer_datetime_format=True)


startDate = '2017-01-21'
endDate = '2020-11-01'

df = df.drop(['device'], axis=1)
df.sort_values(by='date', inplace=True, ascending=False)
df.text = df.text.str.lower()
df = df[ df['date'] > '2017-01-21 00:00:00'] # Only considering from inaguartion date (January 21, 2017) to November 6, 2020 (The day this data was colelcted)
df.drop(['isRetweet', 'isDeleted'], axis=1, inplace=True)
# df = df[ df['favorites'] > 300000 ]

# Import Stock Data
dfSP = pd.read_csv(filepath_or_buffer='S&P500.csv')
dfSP.Security = dfSP.Security.str.lower()
securityArray = dfSP.Security.tolist()




# Check if tweet contains the name of an S&P company
npText = df['text'].to_numpy()
npSecurity = np.array(securityArray)
results = []
tweetedsecurity = []
times = []
count=0
# for i in range(len(npSecurity)):
#     results.append(list(filter(lambda x: npSecurity[i] in x, npText)))

for i in range(len(npSecurity)): # Removed Security's such as 'Visa', 'Ball', 'PPL', and 'GAP' because they can be commonly mistaken for their non-company counterparts
    for j in range(len(df)):
        if(npSecurity[i] in npText[j]):
            results.append(npText[j])
            count+=1
            tweetedsecurity.append(npSecurity[i])
            times.append(df['date'].iloc[j])

# print(results, count, tweetedsecurity)





# create empty list to store ticker symbols in
tickers = []

# loop to fill out the ticker list
for i in range(len(tweetedsecurity)):
    for j in range(len(dfSP['Security'])):
        if(dfSP['Security'].iloc[j] == tweetedsecurity[i]):
            tickers.append(dfSP['Symbol'].iloc[j])



# create empty list to store prices of stocks at time (or some amount of time after? can change this) of tweets
prices = []

# this loop fills out the 'prices' list defined above
for i in range(len(tickers)):
    current_stock = tickers[i]
    current_date = times[i]
    current_stock_info = web.DataReader(current_stock,'yahoo',current_date,current_date)
    current_price = current_stock_info['Close']
    prices.append(current_price)

# start sentiment analysis process
nlp = spacy.load("en_core_web_sm")

token_all = []
for i in range(len(results)):
    doc = nlp(results[i])
    token_list = [token for token in doc]
    filtered_tokens = [token for token in doc if not token.is_stop]
    token_all.append(filtered_tokens)