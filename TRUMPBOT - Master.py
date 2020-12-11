import numpy as np
import pandas as pd
import datetime
import spacy
import os
import random
from spacy.util import minibatch, compounding
import yfinance as yf
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

## IMPORT AND CLEAN TRUMPS TWEETS ##
df = pd.read_csv(filepath_or_buffer='/Users/alec/Desktop/PythonforFinanceCode/FinalProject/trumpTweets.csv', parse_dates=True, infer_datetime_format=True)

startDate = '2017-01-21'
endDate = '2020-11-01'

df = df.drop(['device'], axis=1)
df = df.sort_values(by='date', ascending=False) 
df.text = df.text.str.lower()
df = df[ df['date'] > '2017-01-21 00:00:00'] # Only considering from inaguartion date (January 21, 2017) to November 6, 2020 (The day this data was colelcted)
df.drop(['isRetweet', 'isDeleted'], axis=1, inplace=True)
print(df)
df.to_csv("TweetssansTime.csv", index=False)

# Import Stock Data
dfSP = pd.read_csv(filepath_or_buffer='/Users/alec/Desktop/PythonforFinanceCode/FinalProject/S&P500.csv')
dfSP.Security = dfSP.Security.str.lower()
securityArray = dfSP.Security.tolist()

# Check if tweet contains the name of an S&P company
npText = df['text'].to_numpy()
npTicker = dfSP['Symbol'].to_numpy()
npTime = df['date'].to_numpy()
npID = df['id'].to_numpy()
npSecurity = np.array(securityArray)
results = []
tweetedsecurity = []
ticker = []
time = []
tweetID = []
count=0

for i in range(len(npSecurity)): # Removed Security's such as 'Visa', 'Ball', 'PPL', 'TWTR', 'TARGET', 'PROGRESSIVE', and 'GAP' because they can be commonly mistaken for their non-company counterparts
    for j in range(len(df)):
        if(npSecurity[i] in npText[j]):
            results.append(npText[j])
            count+=1
            tweetedsecurity.append(npSecurity[i])
            ticker.append(npTicker[i])
            time.append(npTime[j])
            tweetID.append(npID[j])

dfParsed = pd.DataFrame([])
dfParsed['Ticker'] = ticker
dfParsed['Company'] = tweetedsecurity
dfParsed['Tweet'] = results
dfParsed['Date'] = time
dfParsed.to_csv("TweetsParsed.csv", index=False) # EXPORT TWEET LIST TO EXCEL TO DO SOME FORMATTING. Formatting was to burdensome to be done in Pandas. For example, rounding to nearest trading day in future

# Import Time Adjsuted Tweets from Computer -- Rounded day of tweet to nearest trading day. If tweet was made after hours, rounded to 9:30 on the following trading day. Done in Excel

df = pd.read_csv(filepath_or_buffer='/Users/alec/Desktop/PythonforFinanceCode/FinalProject/CleanedTimeDate.csv', parse_dates=True, infer_datetime_format=True)
print(df)




## COLLECT AND MERGE STOCK DATA ##
# Import Time Adjsuted Tweets from Computer -- Rounded day of tweet to nearest trading day. If tweet was made after hours, rounded to 9:30 on the following trading day. Done in Excel
df = pd.read_csv(filepath_or_buffer='/Users/alec/Desktop/PythonforFinanceCode/FinalProject/CleanedTimeDate.csv', parse_dates=True, infer_datetime_format=True)


# Goes through all tickers, pull's Open and Close data on that ticker for the correlated day. Wanted to collect data in the 15min period post tweet, but Yahoo Finance only posts minute by minute data in the last 60 days
tickers = df['Ticker'].to_numpy()
openArr = []
closeArr = []
for i in range(len(tickers)):
    tempOpen = []
    tempClose = []

    tempOpen = (yf.download(tickers = tickers[i], start=df['RoundDate'][i])['Open'])
    openArr.append(tempOpen[0])

    tempClose = (yf.download(tickers = tickers[i], start=df['RoundDate'][i])['Adj Close'])
    closeArr.append(tempClose[0])

print(openArr)
print(closeArr)

df["Open"] = openArr
df["Close"] = closeArr
df['PctChange'] = (df["Close"] - df["Open"])/df["Open"]

df.to_csv('tweetsWithOpenClose.csv', index=False)




## SENTIMENT ANALYSIS ##
## DISCLAIMER -- We take no credit for the sentiment analysis code. Deep learning was above our capabilities but necessary for the project. All credit goes to Kyle Stratis at realpython.com

dfParsed = pd.read_csv('tweetsWithOpenClose.csv')
print(dfParsed)

# pd.set_option("display.max_rows", None, "display.max_columns", None)

## Four stages for sentiment analysis
# 1. Load Data
# 2. Preprocessing
# 3. Training the classifier
# 4. Classifying data

performanceStatistics = []


# Takes all files and loads them into one dataset - Using an IMBD review dataset containing > 25,000 reviews
def load_training_data(
    data_directory: str = "/Users/alec/Desktop/PythonforFinanceCode/FinalProject/movieTrainingData/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label))

# Randomly shuffles data - Removes any timeframe correlation betwen different reviews
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

def evaluate_model(
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]["cats"]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")

def test_model(TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(TEST_REVIEW)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {TEST_REVIEW}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
        
    )
    return(prediction)


print(dfParsed)
dfParsed["Sentiment"] = "N/A"


# Run Donald Trumps tweets through the natural language processor model built above
if __name__ == "__main__":
    for i in range(len(dfParsed)):
        TEST_REVIEW = dfParsed["Tweet"][i]
        print(TEST_REVIEW)
        dfParsed["Sentiment"][i] = test_model(TEST_REVIEW)
        
# Export to CSV
dfParsed.to_csv('FinalCSV.csv', index=False)




## LINEAR REGRESSION ##

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