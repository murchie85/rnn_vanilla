import numpy as np



# Prepare your data
# Generate some data
X = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])
y = np.array([[0.4],
              [0.7],
              [1.0]])


# Define the RNN model
# Set the number of units in the hidden layer
num_hidden_units = 4

# Initialize the weight matrices and bias vectors for the RNN
Wxh = np.random.randn(num_hidden_units, 1)
Whh = np.random.randn(num_hidden_units, num_hidden_units)
Why = np.random.randn(1, num_hidden_units)
bh = np.zeros((num_hidden_units, 1))
by = np.zeros((1, 1))




# Define the forward pass function
def rnn_forward(X, Wxh, Whh, Why, bh, by):
    # Initialize the hidden state to zero
    h = np.zeros((num_hidden_units, 1))

    # Initialize the output sequence
    Y = np.zeros((X.shape[0], 1))

    # Loop over each time step in the input sequence
    for t in range(X.shape[0]):
        # Get the current input and reshape it to a column vector
        x = X[t].reshape((-1, 1))

        # Compute the new hidden state
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

        # Compute the output at the current time step
        y = np.dot(Why, h) + by

        # Add the output to the output sequence
        Y[t] = y

    return Y

# Define the loss function
def mse_loss(Y, y_true):
    return np.mean((Y - y_true) ** 2)



# Set the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 100

# Train the RNN
for epoch in range(num_epochs):
    # Perform the forward pass
    Y_pred = rnn_forward(X, Wxh, Whh, Why, bh, by)

    # Compute the loss
    loss = mse_loss(Y_pred, y)

    # Compute the gradients using backpropagation
    dLdY = 2 * (Y_pred - y)
    dLdWhy = np.dot(dLdY.T, h.T)
    dLdh = np.dot(dLdY, Why)
    dLdZ = dLdh * (1 - h ** 2)
    dLdWhh = np.dot(dLdZ.T, h_prev.T)
    dLdWxh = np.dot(dLdZ.T, x.T)
    dLdbh = np.sum(dLdZ, axis=1, keepdims=True)
    dLdby =
