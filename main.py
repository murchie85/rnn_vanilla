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


# Wxh, Whh, Why, bh, and by are the weight matrices and bias vectors that define the parameters of your RNN model. Specifically,
# Wxh is the weight matrix that connects the input X to the hidden state h.
# Whh is the weight matrix that connects the previous hidden state h to the current hidden state h.
# Why is the weight matrix that connects the hidden state h to the output Y.
# bh is the bias vector for the hidden state h.
# by is the bias vector for the output Y.

Wxh = np.random.randn(X.shape[1], num_hidden_units)
Whh = np.random.randn(num_hidden_units, num_hidden_units)
Why = np.random.randn(num_hidden_units,1)
bh = np.zeros((num_hidden_units, 1))
by = np.zeros((1, 1))


# Define the forward pass function to return the output sequence and final hidden state
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

    return Y, h



# Define the loss function
def mse_loss(Y, y_true):
    return np.mean((Y - y_true) ** 2)

# Set the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 100

"""
dLdY is the derivative of the loss with respect to the output sequence Y. It is computed as dLdY = 2 * (Y_pred - y), where Y_pred is the predicted output sequence and y is the true output sequence. The factor of 2 comes from the chain rule of differentiation.

dLdWhy is the derivative of the loss with respect to the weight matrix Why. It is computed as dLdWhy = np.dot(dLdY.T, h).T, where h is the hidden state at the last time step.

dLdh is the derivative of the loss with respect to the hidden state h. It is computed as dLdh = np.dot(Why.T, dLdY).

dLdZ is the derivative of the loss with respect to the input to the activation function for the hidden state. It is computed as dLdZ = dLdh * (1 - h**2), where (1 - h**2) is the derivative of the hyperbolic tangent activation function.

dLdWhh is the derivative of the loss with respect to the weight matrix Whh. It is computed as dLdWhh = np.dot(dLdZ, h.T).

dLdWxh is the derivative of the loss with respect to the weight matrix Wxh. It is computed as dLdWxh = np.dot(dLdZ, X).

dLdbh is the derivative of the loss with respect to the bias vector bh. It is computed as dLdbh = np.sum(dLdZ, axis=1, keepdims=True).

dLdby is the derivative of the loss with respect to the bias vector by. It is computed as dLdby = np.sum(dLdY, axis=0, keepdims=True).

"""



# Train the RNN
for epoch in range(num_epochs):
    # Perform the forward pass
    Y_pred, h = rnn_forward(X, Wxh, Whh, Why, bh, by)

    # Compute the loss
    loss = mse_loss(Y_pred, y)

    # Compute the gradients using backpropagation
    dLdY = 2 * (Y_pred - y).T
    print(dLdY)

    #dLdWhy = np.dot(dLdY.T, h.T)
    #dLdWhy = np.dot(dLdY.T, h).T
    #dLdWhy = np.dot(dLdY.T, h).T
    dLdWhy = np.dot(h, dLdY).T
    dLdh = np.dot(dLdY, Why)
    dLdZ = dLdh * (1 - h ** 2)
    dLdWhh = np.dot(dLdZ.T, h.T)
    dLdWxh = np.dot(dLdZ.T, X).T
    dLdbh = np.sum(dLdZ, axis=1, keepdims=True)
    dLdby = np.sum(dLdY, axis=0, keepdims=True)

    # Update the weights and biases using SGD
    Wxh -= learning_rate * dLdWxh
    Whh -= learning_rate * dLdWhh
    Why -= learning_rate * dLdWhy.T
    bh -= learning_rate * dLdbh
    by -= learning_rate * dLdby

    # Print the loss every 10 epochs
    if epoch % 10 == 0:
        print('Epoch %d, Loss: %f' % (epoch, loss))

# Generate a new sequence
seq_len = X.shape[0]
h = np.zeros((num_hidden_units, 1))
Y_pred, _ = rnn_forward(X, Wxh, Whh, Why, bh, by)

for t in range(seq_len, seq_len + 10):
    # Get the current input and reshape it to a column vector
    x = np.array([[X[t - 1]]])

    # Compute the new hidden state
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

    # Compute the output at the current time step
    y = np.dot(Why, h) + by

    # Add the output to the output sequence
    Y_pred = np.vstack((Y_pred, y))

# Print the output sequence
print('Output sequence:')
print(Y_pred)

