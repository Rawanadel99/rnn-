#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Prepare the data
data = "dogs"
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Set dimensions
input_size = vocab_size   
hidden_size = 3            
output_size = vocab_size

# Initialize weights randomly
np.random.seed(42)
Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden
Why = np.random.randn(output_size, hidden_size) * 0.01 # hidden to output
bh = np.zeros((hidden_size, 1))                        # hidden bias
by = np.zeros((output_size, 1))                        # output bias

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Forward and backward pass
def lossFun(inputs, targets, hprev):
    """
    inputs: list of input character indices
    targets: list of target character indices
    hprev: initial hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # Forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((input_size, 1))  # one-hot encoding
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = softmax(ys[t])
        loss += -np.log(ps[t][targets[t], 0])  # cross-entropy loss

    # Backward pass: compute gradients
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # tanh derivative
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

inputs = [char_to_ix[ch] for ch in data[:-1]]  # d,o,g
targets = [char_to_ix[ch] for ch in data[1:]]  # o,g,s

hprev = np.zeros((hidden_size, 1))


loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)


print("Loss:", loss)
print("\nGradient dWxh:\n", dWxh)
print("\nGradient dWhh:\n", dWhh)
print("\nGradient dWhy:\n", dWhy)
print("\nGradient dbh:\n", dbh)
print("\nGradient dby:\n", dby)


# In[ ]:




