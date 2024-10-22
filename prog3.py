import pandas as pd
import numpy as np

# Load dataset
dataset = pd.read_csv('playtennis.csv', names=['outlook', 'temperature', 'humidity', 'wind', 'class'])

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])

# Function to calculate Information Gain
def info_gain(data, split_attr, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attr], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attr] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    return total_entropy - weighted_entropy

# ID3 Algorithm
def ID3(data, original_data, features, target_name="class", parent_node_class=None):
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_name])[np.argmax(np.unique(original_data[target_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
        best_feature = features[np.argmax([info_gain(data, feature, target_name) for feature in features])]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, original_data, features, target_name, parent_node_class)
            tree[best_feature][value] = subtree
        return tree

# Prediction function
def predict(query, tree, default=1):
    for key in query.keys():
        if key in tree.keys():
            try:
                result = tree[key][query[key]]
            except:
                return default
            return predict(query, result) if isinstance(result, dict) else result

# Split data into training
def train_test_split(dataset):
    return dataset.iloc[:14].reset_index(drop=True)

# Test function
def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)
    accuracy = (np.sum(predicted["predicted"] == data["class"]) / len(data)) * 100
    print(f'The prediction accuracy is: {accuracy}%')

# Train and test the model
training_data = train_test_split(dataset)
tree = ID3(training_data, training_data, training_data.columns[:-1])
print('Decision Tree:', tree)
test(training_data, tree)
