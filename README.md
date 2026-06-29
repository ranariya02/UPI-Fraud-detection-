# UPI Fraud Detection using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    'amount': [100, 5000, 200, 15000, 300, 25000, 400, 10000],
    'transaction_time': [10, 23, 14, 2, 11, 1, 15, 3],
    'location_change': [0, 1, 0, 1, 0, 1, 0, 1],
    'fraud': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['amount', 'transaction_time', 'location_change']]
y = df['fraud']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# New Transaction Check
new_transaction = [[12000, 2, 1]]
prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("⚠ Fraudulent Transaction Detected")
else:
    print("✓ Genuine Transaction")# UPI-Fraud-detection-
