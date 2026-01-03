# **Dogecoin Price Prediction via NLP**

Project utilizes NLP to predict Dogecoin price movements based on social media sentiment.

## **Core Files**

LR_Simple.py: Main script using Logistic Regression for price classification.

ARIMA_Rolling.py: Time-series forecasting for price trends.

SentBERT+ARIMA Data Generator.py: Pipeline combining BERT embeddings with ARIMA data.

WordCount.py & filter_file.py: Scripts for data cleaning and keyword analysis.

## **Data & Lexicons**

dogecoin_prices.csv: Historical price data.

2024_dogecoin_posts.csv: Social media post samples.

crypto_general_positive/negative_lexicon.txt: Crypto-specific sentiment dictionaries.

cry_word_counts.csv: Frequency data used for feature engineering.

Note: Raw comments data was too large to upload. Programs not used to generate final results are omitted.

## **Setup**

Install requirements: pandas, scikit-learn, statsmodels, sentence-transformers.

Run LR_Simple.py for final prediction results.
