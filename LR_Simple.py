import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
import requests
import random

#Load data from csv file
input_file = r"C:\Users\johnn\Downloads\cry_word_counts.csv"
df = pd.read_csv(input_file, header=None)
df.columns = ['embedding', 'label']
def parse(row):
    # Remove brackets and split on space
    numbers = row.split()

    # Convert to floats
    numbers = [float(x) for x in numbers]
    return numbers

# Apply the function to the column
parsed = df['embedding'].apply(parse)

X_train = torch.tensor(parsed.tolist())
y_train = torch.tensor(df['label'].values)

# hyperparameters
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
lr = 0.00001
n_epochs = 50
n_examples = X_train.shape[0]
n_feats = X_train.shape[1]
n_classes = 3

# initialize the model, loss function, and optimizer
model = nn.Sequential(
    nn.Linear(n_feats, n_classes)
)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

#train the model
indices = np.arange(n_examples)
for epoch in range(n_epochs):
    n_errors = 0
    # randomize training examples
    np.random.shuffle(indices)
    # for each training example
    for i in tqdm(indices, desc=f'epoch {epoch+1}'):
        x = X_train[i].unsqueeze(0)
        y_true = y_train[i].unsqueeze(0)
        # make predictions
        y_pred = model(x)
        # calculate loss
        loss = loss_func(y_pred, y_true)
        # calculate gradients through back-propagation
        loss.backward()
        # optimize model parameters
        optimizer.step()
        # ensure gradients are set to zero
        model.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# set model to evaluation mode
model.eval()
test_file = r"C:\Users\johnn\Downloads\test_cry_word_counts.csv"
testdf = pd.read_csv(test_file, header=None)
testdf.columns = ['embedding', 'label']

# Apply the function to the test column
parsed = testdf['embedding'].apply(parse)
X_test = torch.tensor(parsed.tolist())
y_test = torch.tensor(testdf['label'].values)

# don't store gradients
with torch.no_grad():
 y_pred = torch.argmax(model(X_test), dim=1)
 y_pred = y_pred.cpu().numpy()
 print(classification_report(y_test, y_pred, target_names=['down', 'no change', 'up']))
'''
performance on sample:
              precision    recall  f1-score   support

        down       0.36      0.32      0.34        60
   no change       0.56      0.39      0.46       165
          up       0.34      0.58      0.43        81

    accuracy                           0.43       306
   macro avg       0.42      0.43      0.41       306
weighted avg       0.46      0.43      0.43       306

performance on test:
              precision    recall  f1-score   support

        down       0.38      0.71      0.50         7
   no change       0.57      0.50      0.53        16
          up       0.33      0.14      0.20         7

    accuracy                           0.47        30
   macro avg       0.43      0.45      0.41        30
weighted avg       0.47      0.47      0.45        30
'''
