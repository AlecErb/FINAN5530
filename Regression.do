import delimited "/Users/alec/Desktop/PythonforFinanceCode/FinalProject/TweetsSentimentPctChange.csv", encoding(UTF-8) clear
 
* Dummy code Sentiment
generate neg = (sentiment=="Negative")



* Clean data
drop if missing(id)
destring pctchange, replace

* Regress
reg pctchange neg, robust
