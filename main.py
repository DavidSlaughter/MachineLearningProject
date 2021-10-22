import numpy
import matplotlib.pyplot
import scipy.special    # For sigmoid function expit()

class neuralNetwork:

    # Initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Link weight matrices, wih and who
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # Learning rate
        self.lr = learningrate

        # Activation function is the sigmoid function
        self.activation_function = lambda x : scipy.special.expit(x)

        pass

    # Train the neural network
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Output layer error is the target - actual
        output_errors = targets - final_outputs
        # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    # Query the neural network
    def query(self, inputs_list):
        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Number of input, hidden, and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# Learning rate is 0.3
learning_rate = 0.15

# Create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Load the mnist training data CSV file into a list
training_data_file = open("/Users/davidjs/Documents/Programming/mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Train the neural network

# Epochs is the number of times the training data set is used for training
epochs = 7

for e in range(epochs):
    # Go through all records in the training data set for record in training_data_list:
    for record in training_data_list:
        # Split the record by the ',' commas
        all_values = record.split(',')
        # Scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # Create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# Loads the mnist test data CSV file into a list
test_data_file = open("/Users/davidjs/Documents/Programming/mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Get the first test record
all_values = test_data_list[0].split(',')
# Print the label
print(all_values[0])

print(n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))


# Test the neural network

# Scorecard for how well the network performs, initially empty
scorecard = []

# Go through all the records in the test data set
for record in test_data_list:
    # Split the record by the ',' commas
    all_values = record.split(',')
    # Correct answer is the first value
    correct_label = int(all_values[0])
    print(f"{correct_label} = 'correct label'")
    # Scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # Query the network
    outputs = n.query(inputs)
    # The index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(f"{label} = 'network\'s answer'")
    # Append correct or incorrect to list
    if (label == correct_label):
        # Network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # Network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

print(scorecard)

# Calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("Performance = ", 100 * scorecard_array.sum() / scorecard_array.size, '%')

