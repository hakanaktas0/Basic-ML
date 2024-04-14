
# Run with the following command:
# python3 logreg.py <num_epochs> norm shuffle bias

import sys
from time import time
import numpy as np
import pandas as pd
from math import e, log
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
  
# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 

# variable information 
# print(rice_cammeo_and_osmancik.variables)

# data (as pandas dataframes) 
x = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
# metadata 
metadata = rice_cammeo_and_osmancik.metadata
num_instances = metadata['num_instances']
num_features = metadata['num_features']

# training parameters
regularization_terms = [0.0, e**(-18), e**(-9), e**(-6), e**(-3)]
num_epochs = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdecimal()) else 1000
default_learning_rate = 0.001

# classes
Y_TABLE = {
    "Osmancik": 1.0,
    "Cammeo":   0.0,
}

# pandas.DataFrame -> NumpyArray
x = x.to_numpy()
y = np.asarray([[Y_TABLE[value]] for value in y.Class])

# Linear normalization to [0,1]
if "norm" in sys.argv:
    print("Normalized...")
    for i in range(num_features):
        min_value = np.min(x[ :, i])
        max_value = np.max(x[ :, i])
        for j in range(num_instances):
            x[j,i] = (x[j,i]-min_value) / (max_value-min_value)

# Shuffle the data set
if "shuffle" in sys.argv:
    print("Shuffled...")
    combined = np.append(x, y, axis=1)
    np.random.shuffle(combined)
    x = combined[ :, :-1]
    y = combined[ :,  -1]

# Add bais term
if "bias" in sys.argv:
    print("Bias is added...")
    x = np.append(x, np.asarray([[1.0] for i in range(num_instances)]), axis=1)

# Generates a random weight vector
def get_random_w(low = -0.5, high = 0.5):
    dim = num_features
    if "bias" in sys.argv:
        dim += 1
    return np.array([(high-low)*np.random.random()+low for i in range(dim)])

# Sigmoid function
def sigmoid(s) -> float:
    return 1.0 / (1.0 + e**(-s))

# Returns 1.0 if sigmoid(wTx) > 0.5, otherwise 0.0
def predict(w, x) -> float:
    return float(sigmoid(w.dot(x)) >= 0.5)

# Loss = SUM(y * ln(sigmoid(wTx)) + (1-y) * ln(1.0-sigmoid(wTx))) + 0.5 * *lmd * wTw
def loss(train_x, train_y, train_w, lmd) -> float:
    size = train_x.shape[0]
    result = lmd/2.0 * train_w.dot(train_w)
    for i in range(size):
        prob = sigmoid(train_w.dot(train_x[i,:]))

        # Avoid any Domain Errors due to log function
        if prob <= 0.0:
            prob = 0.0000001
        elif prob >= 1.0:
            prob = 0.9999999 

        result -= train_y[i]*log(prob) + (1.0-train_y[i])*log(1.0-prob)
    return result

# Tests the weights on a given test set
# Returns the percent of hits
def test(test_x, test_y, test_w) -> float:
    test_size = test_x.shape[0]
    num_hits = 0
    for i in range(test_size):
        pred = predict(test_w, test_x[i,:])
        if (pred == test_y[i]):
            num_hits += 1
    return num_hits / test_size * 100.0

# Runs GD once and updates train_w
def gradient_descent(train_x, train_y, train_w, lr, lmd):
    train_size = train_x.shape[0]
    gradient = sum([(sigmoid(train_w.dot(train_x[i,:])) - train_y[i]) * train_x[i,:] for i in range(train_size)]) + lmd*train_w
    train_w -= lr * gradient

# Runs SGD once and updates train_w
def stochastic_gradient_descent(train_x, train_y, train_w, lr, lmd):
    train_size = train_x.shape[0]
    for i in range(train_size):
        gradient = (sigmoid(train_w.dot(train_x[i,:])) - train_y[i]) * train_x[i,:] + lmd*train_w
        train_w -= lr * gradient

# Trains and tests a model
# Prints the train time and accuracy
def simple_train(num_epochs, lmd, gd=True, label=""):
    # First 20% of data is the test set
    # Remaining 80% is the train set
    block_size = num_instances // 5
    test_x = x[:block_size,:]
    test_y = y[:block_size]
    train_x = x[block_size:,:]
    train_y = y[block_size:]

    start_time = time()
    w = get_random_w()
    for epoch in range(num_epochs):
        if gd:
            gradient_descent(train_x, train_y, w, default_learning_rate, lmd)
        else:
            stochastic_gradient_descent(train_x, train_y, w, default_learning_rate, lmd)
    
    model = "GD" if gd else "SGD"
    print("%s %s train time: %.2f s" % (model, label, time()-start_time))
    train_accuracy = test(train_x, train_y, w)
    print("%s %s train accuracy: %.2f %%" % (model, label, train_accuracy))
    test_accuracy = test(test_x, test_y, w)
    print("%s %s test accuracy: %.2f %%" % (model, label, test_accuracy))

# Runs SGD for different step sizes, and plots the loss with respect to epoch
def compare_learning_rates(num_epochs, learning_rates, lmd):
    # Last 80% percent of the data is the train data
    block_size = num_instances // 5
    train_x = x[block_size:,:]
    train_y = y[block_size:]

    for lr in learning_rates:
        train_w = get_random_w()
        losses = []
        for epoch in range(num_epochs):
            stochastic_gradient_descent(train_x, train_y, train_w, lr, lmd)
            losses.append(loss(train_x, train_y, train_w, lmd))
        plt.plot(losses, label="Learning Rate %.4f" % lr)

    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Comparison of Different Learning for SGD")
    plt.legend()
    plt.show()

# Trains a GD model and a SGD model for a target accuracy
# Plots the losses with respect to epoch
def get_training_time(gd_lmd, sgd_lmd, target_accr):
    # First 20% of data is the test set
    # Remaining 80% is the train set
    block_size = num_instances // 5
    test_x = x[:block_size,:]
    test_y = y[:block_size]
    train_x = x[block_size:,:]
    train_y = y[block_size:]

    gd_w = get_random_w()
    gd_losses = [loss(train_x, train_y, gd_w, gd_lmd)] # Untrained loss
    gd_accrs = [test(test_x, test_y, gd_w)] # Untrained accuracy
    gd_count = 0
    print("GD accuracy step %d: %.2f" % (gd_count, gd_accrs[-1]))
    
    start_time = time()
    while gd_accrs[-1] < target_accr:
        gradient_descent(train_x, train_y, gd_w, default_learning_rate, gd_lmd)
        gd_losses.append(loss(train_x, train_y, gd_w, gd_lmd))
        gd_accrs.append(test(test_x, test_y, gd_w))
        print("GD accuracy step %d: %.2f" % (gd_count, gd_accrs[-1]))
        gd_count += 1
    gd_train_time = time() - start_time

    sgd_w = get_random_w()
    sgd_losses = [loss(train_x, train_y, sgd_w, sgd_lmd)] # Untrained loss
    sgd_accrs = [test(test_x, test_y, sgd_w)] # Untrained accuracy
    sgd_count = 0
    print("SGD accuracy step %d: %.2f" % (sgd_count, sgd_accrs[-1]))
    
    start_time = time()
    while sgd_accrs[-1] < target_accr:
        stochastic_gradient_descent(train_x, train_y, sgd_w, default_learning_rate, sgd_lmd)
        sgd_losses.append(loss(train_x, train_y, sgd_w, sgd_lmd))
        sgd_accrs.append(test(test_x, test_y, sgd_w))
        print("SGD accuracy step %d: %.2f" % (sgd_count, sgd_accrs[-1]))
        sgd_count += 1
    sgd_train_time = time() - start_time

    print("GD needed %.2f s and %d epochs to reach %.2f %% accuracy" % (gd_train_time, gd_count, target_accr))
    print("SGD needed %.2f s and %d epochs to reach %.2f %% accuracy" % (sgd_train_time, sgd_count, target_accr))

    plt.plot(gd_losses, label="GD")
    plt.plot(sgd_losses, label="SGD")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss With Respect to Epoch")
    plt.legend()
    plt.show()

    plt.plot(gd_accrs, label="GD")
    plt.plot(sgd_accrs, label="SGD")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy %")
    plt.title("Accuracy With Respect to Epoch")
    plt.legend()
    plt.show()

# Returns the best regularization terms for GD and SGD
def get_best_regularization_term(num_epochs, terms):
    # Split the data into <fold> many blocks
    fold = len(terms)
    block_size = num_instances // fold
    x_blocks = [x[start:(start+block_size),:] for start in range(0, num_instances, block_size)]
    y_blocks = [y[start:(start+block_size)]   for start in range(0, num_instances, block_size)]

    #  Statistics
    gd_stats = [0.0 for i in range(fold)]
    sgd_stats = [0.0 for i in range(fold)]

    for i, lmd in enumerate(terms):
        for index in range(fold):
            # Determine the test and train sets
            test_x = x_blocks[index]
            test_y = y_blocks[index]
            train_x = np.concatenate([x_blocks[i] for i in range(fold) if i != index], axis=0)
            train_y = np.concatenate([y_blocks[i] for i in range(fold) if i != index], axis=0)
            
            # Train and test the model for GD
            gd_w = get_random_w()
            for epoch in range(num_epochs):
                gradient_descent(train_x, train_y, gd_w, default_learning_rate, lmd)
            accuracy = test(test_x, test_y, gd_w) 
            gd_stats[i] += accuracy

            # Train and test the model for SGD
            sgd_w = get_random_w()
            for epoch in range(num_epochs):
                stochastic_gradient_descent(train_x, train_y, sgd_w, default_learning_rate, lmd)
            accuracy = test(test_x, test_y, sgd_w)
            sgd_stats[i] += accuracy
        print("Term %d is completed" % (i+1))

    gd_stats = [accr/fold for accr in gd_stats]
    sgd_stats = [accr/fold for accr in sgd_stats]
    gd_lmd = terms[gd_stats.index(max(gd_stats))]
    sgd_lmd = terms[sgd_stats.index(max(sgd_stats))]

    print("GD average accuracies", gd_stats)
    print("SGD average accuracies", sgd_stats)
    print("GD ln(lambda) = ", log(gd_lmd) if gd_lmd != 0.0 else float("-inf"))
    print("SGD ln(lambda) = ", log(sgd_lmd) if sgd_lmd != 0.0 else float("-inf"))
    return gd_lmd, sgd_lmd


# Train GD and SGD
simple_train(num_epochs, 0.0, gd=True)
simple_train(num_epochs, 0.0, gd=False)

# Determine the best regularization term for GD and SGD
# gd_lmd, sgd_lmd = get_best_regularization_term(num_epochs, regularization_terms)

# Compare GD (lmd=0), GD (lmd=e^(-18)), SGD (lmd=0), SGD (lmd=e^(-18)), 
# simple_train(num_epochs, 0.0, gd=True)
# simple_train(num_epochs, e**(-18), gd=True)
# simple_train(num_epochs, 0.0, gd=False)
# simple_train(num_epochs, e**(-18), gd=False)

# Calculate the required time to reach the target accuracy
# target_accr = 92.0 # percent
# get_training_time(e**(-18), e**(-18), target_accr)

# Compare different step sizes on SGD
# learning_rates = [5.0, 4.0, 3.0, 2.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
# compare_learning_rates(num_epochs, learning_rates, e**(-18))

