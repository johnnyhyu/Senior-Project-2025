import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import requests
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load CSV file
input_file = r"C:\Users\johnn\Downloads\2024_dogecoin_posts.csv"

df = pd.read_csv(input_file, header=None)

# Load the SentBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Concatenate text columns
df['combined_text'] = df[2].astype(str) + " " + df[5].astype(str)

# Compute embeddings for concatenated text
df['embedding'] = df['combined_text'].apply(lambda text: model.encode(text))

# Average embeddings for each day
average_embeddings = df.groupby(df[1])['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))

#Load pricing data
price_file = r"C:\Users\johnn\Downloads\dogecoin_prices.csv"
pricedf = pd.read_csv(price_file, index_col=False)
pricedf.set_index("timestamp", inplace=True)


# Rolling ARIMA Model with 30-Day Window & AIC Optimization
train_window = 30  # Number of days used for training

# Grid of (p, d, q) values to test
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

# Function to find best ARIMA order using AIC
def find_best_arima(train_data):
    best_aic = float("inf")
    best_order = None
    best_model = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = sm.tsa.ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except:
                    continue

    return best_model, best_order

# Iterate over the test period, rolling 1 day at a time
c = train_window
for i in range(train_window, len(pricedf) - 1):
    train_data = pricedf.iloc[i - train_window:i]  # Last 30 days

    try:
        # Find best ARIMA model based on AIC
        best_model, best_order = find_best_arima(train_data)
        pred = best_model.forecast(steps=1)[0]
        #add ARIMA to embedding
        average_embeddings[c] = np.append(average_embeddings[c], pred - pricedf['price'][c - 1])
        c = c + 1
    except Exception as e:
        print(f"Error on day {pricedf.index[i]}: {e}")

# Save result to a file
output_file = r"C:\Users\johnn\Downloads\average_embeddings.csv"
average_embeddings.to_csv(output_file, index=False)
