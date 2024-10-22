from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
y = iris.target  # Target classes (0, 1, 2 representing species)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize the k-Nearest Neighbors classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Step 4: Train the k-NN model
knn.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 6: Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of k-NN model: {accuracy * 100:.2f}%")

# Step 7: Print both correct and wrong predictions
print("\nCorrect predictions:")
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        print(f"Sample {i}: True Label = {y_test[i]}, Predicted Label = {y_pred[i]}")

print("\nWrong predictions:")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f"Sample {i}: True Label = {y_test[i]}, Predicted Label = {y_pred[i]}")
