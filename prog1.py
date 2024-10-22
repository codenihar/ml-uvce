import csv

# Function to load the CSV file
def loadCsv(filename):
    # Open and read the CSV file
    lines = csv.reader(open(filename, "rt"))
    # Convert lines into a dataset
    dataset = list(lines)
    
    # Return the dataset as a list
    return dataset

# Attributes list for the problem
attributes = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']
print("Attributes:", attributes)

# Number of attributes
num_attributes = len(attributes)

# CSV file containing weather data
filename = "weather.csv"  # Make sure this file is saved in the same directory
dataset = loadCsv(filename)
print("\nDataset:", dataset)

# Target values (outcomes for each instance)
target = ['Yes', 'Yes', 'No', 'Yes']
print("\nTarget:", target)

# Initial hypothesis (most specific, starting with '0' for all attributes)
hypothesis = ['0'] * num_attributes
print("\nInitial Hypothesis:", hypothesis)

# Find-S algorithm
print("\nThe Hypothesis are:")
for i in range(len(target)):
    if target[i] == 'Yes':  # Only consider positive examples ('Yes')
        for j in range(num_attributes):
            if hypothesis[j] == '0':  # If attribute is unset, set it to current value
                hypothesis[j] = dataset[i][j]
            elif hypothesis[j] != dataset[i][j]:  # If it doesn't match, generalize with '?'
                hypothesis[j] = '?'
        print(f"{i+1} = {hypothesis}")

# Print the final hypothesis after processing all examples
print("\nFinal Hypothesis:")
print(hypothesis)
