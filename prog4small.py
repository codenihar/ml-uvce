import numpy as np
from sklearn.neural_network import MLPClassifier

# Define the dataset
dataset = np.array([[2.781, 2.550, 0], [1.465, 2.362, 0], [3.396, 4.400, 0], [1.388, 1.850, 0], 
                    [3.064, 3.005, 0], [7.627, 2.759, 1], [5.332, 2.089, 1], [6.922, 1.771, 1], 
                    [8.675, -0.242, 1], [7.673, 3.509, 1]])

# Split the data into features (X) and labels (y)
X, y = dataset[:, :-1], dataset[:, -1]

# Initialize and train the MLP model
model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.5, max_iter=20, random_state=1)
model.fit(X, y)

# Test the trained model
predictions = model.predict(X)
accuracy = np.mean(predictions == y) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Display trained weights
for i, layer_weights in enumerate(model.coefs_):
    print(f'Layer {i+1} weights:\n', layer_weights)
