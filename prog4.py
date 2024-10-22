import math
import random

# Initialize the network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    network.append([{'weights': [random.random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)])
    network.append([{'weights': [random.random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)])
    return network

# Calculate neuron activation
def activate(weights, inputs):
    return sum(w * i for w, i in zip(weights[:-1], inputs)) + weights[-1]

# Sigmoid function
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))

# Forward propagate
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)  # Ensure 'output' key is set
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Derivative of Sigmoid
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagation
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i == len(network) - 1:  # Output layer
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])
        else:  # Hidden layers
            for j, neuron in enumerate(layer):
                error = sum(n['weights'][j] * n['delta'] for n in network[i + 1])
                errors.append(error)
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update weights
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1] if i == 0 else [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']  # Bias update

# Train the network
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for _ in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(f'>epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}')

# Test backpropagation
random.seed(1)
dataset = [[2.781,2.550,0], [1.465,2.362,0], [3.396,4.400,0], [1.388,1.850,0], [3.064,3.005,0],
           [7.627,2.759,1], [5.332,2.089,1], [6.922,1.771,1], [8.675,-0.242,1], [7.673,3.509,1]]
n_inputs, n_outputs = len(dataset[0]) - 1, len(set(row[-1] for row in dataset))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)

# Display trained weights
for layer in network:
    print(layer)
