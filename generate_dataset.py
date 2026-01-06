
import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 5000

# Generate synthetic data
data = {
    'Age': np.random.randint(18, 70, size=num_samples),
    'Budget': np.random.randint(500, 5000, size=num_samples),
    'Interet': np.random.choice(['Culture', 'Nature', 'Aventure', 'Plage', 'Ville'], size=num_samples),
    'Duree': np.random.randint(2, 30, size=num_samples),
    'Climat': np.random.choice(['Chaud', 'Froid', 'Tempéré'], size=num_samples),
    'Destination': np.random.choice(['Paris', 'Tokyo', 'New York', 'Bali', 'Rome', 'Le Caire', 'Rio de Janeiro', 'Sydney', 'Barcelone', 'Londres'], size=num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('tourisme_dataset.csv', index=False)

print("Dataset 'tourisme_dataset.csv' created successfully.")
