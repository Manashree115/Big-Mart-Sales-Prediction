from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('Bigbazaar_sales_data.csv')

# Specify the size of the testing set
test_size = 0.2

# Shuffle the data randomly
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df_shuffled, test_size=test_size)

# Print the first few rows of the training and testing sets to verify that the data was split correctly
print('Training data:')
print(train_data.head())
print('Testing data:')
print(test_data.head())

# Save the training and testing sets to separate CSV files without changing the original dataset
train_data.to_csv('Train.csv', index=False)
test_data.to_csv('Test.csv', index=False)
