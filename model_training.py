
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the processed data
df = pd.read_csv('processed_data.csv')

# Separate features and target
X = df.drop('Destination', axis=1)
y = df['Destination']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'recommendation_model.joblib')

print("Model training complete. Model saved to 'recommendation_model.joblib'.")
