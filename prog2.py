import numpy as np
import pandas as pd

# Loading Data from a CSV File
data = pd.DataFrame(pd.read_csv('training_examples.csv'))

# Separating concept features from the Target column
concepts = np.array(data.iloc[:, 0:-1])  # Features (all columns except the last)
target = np.array(data.iloc[:, -1])      # Target (last column)

# Candidate Elimination Algorithm function
def learn(concepts, target):
    '''learn() function implements the learning method of the Candidate Elimination algorithm.
    Arguments:
    concepts - a NumPy array with all the features (concepts from the dataset)
    target   - a NumPy array with corresponding output values (target labels)
    '''
    
    # Initialize S0 with the first instance from concepts (specific hypothesis)
    specific_h = concepts[0].copy()
    print("Initialization of Specific Hypothesis (S0) and General Hypothesis (G0):")
    print("Specific Hypothesis:", specific_h)

    # Initialize General Hypothesis G0 to the most general hypothesis (all '?')
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("General Hypothesis:", general_h)

    # Iterate over all training instances
    for i, h in enumerate(concepts):
        # If the target is positive, generalize the specific hypothesis
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'  # Generalize specific_h
                    general_h[x][x] = '?'  # Generalize general_h

        # If the target is negative, specialize the general hypothesis
        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]  # Specialize general_h
                else:
                    general_h[x][x] = '?'  # Retain generality if the attribute matches
                
        print(f"\nSteps of Candidate Elimination Algorithm {i + 1}:")
        print("Specific Hypothesis:", specific_h)
        print("General Hypothesis:", general_h)
    
    # Remove overly general hypotheses from the final general hypothesis set
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    # Return final specific and general hypotheses
    return specific_h, general_h

# Running the Candidate Elimination algorithm
s_final, g_final = learn(concepts, target)

# Final Output
print("\nFinal Specific Hypothesis:")
print(s_final)
print("\nFinal General Hypothesis:")
print(g_final)
