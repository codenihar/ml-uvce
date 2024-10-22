import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Create Dummy Data
# We will use a simplified dataset with features: [Age, Cholesterol, BloodPressure, Smoking]
# Target variable: HeartDisease (1 = Yes, 0 = No)

X = np.array([[45, 230, 120, 1],  # Age, Cholesterol, Blood Pressure, Smoking (1 = Yes, 0 = No)
              [50, 250, 130, 0],
              [55, 210, 140, 1],
              [60, 220, 150, 1],
              [40, 180, 110, 0],
              [65, 260, 160, 1],
              [35, 190, 115, 0]])

# Target: Whether the person has heart disease or not (1 = Yes, 0 = No)
y = np.array([1, 1, 1, 1, 0, 1, 0])

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize the Naive Bayes Model
model = GaussianNB()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 6: Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Step 7: Predict on new data
new_data = np.array([[50, 200, 125, 0]])  # Input new patient data
prediction = model.predict(new_data)

if prediction == 1:
    print("The model predicts that the patient has heart disease.")
else:
    print("The model predicts that the patient does not have heart disease.")
