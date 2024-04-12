import pandas as pd
import random
import numpy as np
import argparse

# Get arguments
parser = argparse.ArgumentParser(description="Naive Bayes Classifier")
parser.add_argument("--data_path", type=str, help="Path to the data file", required=True)
parser.add_argument("--train_proportion", type=float, help="Proportion of the dataset to include in the training set. Default is 0.8", default=0.8)
parser.add_argument("--seed", type=int, help="Seed for random number generator.", default=None)
parser.add_argument("--calculate_parameter_count", action="store_true", help="Calculate the number of parameters in the model. Default is False", default=False)
parser.add_argument("--repeat", type=int, help="Number of times to repeat the experiment. Default is 1", default=1)
args = parser.parse_args()

# Load the data
data_path = args.data_path
df = pd.read_csv(data_path, sep=",", header=None) # Read the data
df = df.drop(columns=[0]) # Drop the id

if args.seed is not None:
    random.seed(args.seed)

# Calculate the number of parameters
if args.calculate_parameter_count:
    num_features = len(df.columns) - 1 # Drop the target column
    num_classes = len(df[1].unique())
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    params_with_independence = 2 * num_classes * num_features
    params_without_independence = num_classes * (num_features + (num_features * (num_features + 1) // 2)) 
    
    print(f"Number of parameters with conditional independence: {params_with_independence}")
    print(f"Number of parameters without conditional independence: {params_without_independence}")
    
def run():
    # Split the data into training and test sets
    n = len(df)
    train_indices = random.sample(range(n), int(args.train_proportion * n))
    test_indices = list(set(range(n)) - set(train_indices))
    train_data = df.iloc[train_indices]
    test_data = df.iloc[test_indices]

    # Omit the target columns
    train_target = train_data[1]
    train_data = train_data.drop(columns=[1])
    train_data.columns = range(train_data.columns.size)

    test_target = test_data[1]
    test_data = test_data.drop(columns=[1])
    test_data.columns = range(test_data.columns.size)

    # Calculate the mean and standard deviation for each feature
    m_means = train_data[train_target == "M"].mean()
    m_stds = train_data[train_target == "M"].std()
    m_prior = sum(train_target == "M") / len(train_target)

    b_means = train_data[train_target == "B"].mean()
    b_stds = train_data[train_target == "B"].std()
    b_prior = sum(train_target == "B") / len(train_target)

    def calculate_likelihood(x, mean, std):
        return -0.5 * np.log(2 * np.pi * std ** 2) - ((x - mean) ** 2 / (2 * std ** 2))
    
    def posterior(row):
        m_likelihoods = [
            calculate_likelihood(row[index], m_means[index], m_stds[index]) for index in range(0, len(row))
        ]
        m_posterior = np.sum(m_likelihoods) + np.log(m_prior)
        
        b_likelihoods = [
            calculate_likelihood(row[index], b_means[index], b_stds[index]) for index in range(0, len(row))
        ]
        b_posterior = np.sum(b_likelihoods) + np.log(b_prior)
        
        return "M" if (m_posterior > b_posterior) else "B"

    # Predict Data
    def predict(data, X):
        predictions = []
        for _, row in data.iterrows():
            prediction = posterior(row)
            predictions.append(prediction)
        accuracy = sum(X == predictions) / len(data[1])
        return accuracy

    train_accuracy = predict(train_data, train_target)
    test_accuracy = predict(test_data, test_target)    
    return train_accuracy, test_accuracy

train_accuracies = []
test_accuracies = []
for _ in range(args.repeat):
    train_accuracy, test_accuracy = run()
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

print(f"Train Accuracy: {np.mean(train_accuracies)}")
print(f"Test Accuracy: {np.mean(test_accuracies)}")