import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import PLA

parser = argparse.ArgumentParser()

parser.add_argument('--data_location',help='please specify the path of the data (including PLA_data folder)',required=True)
parser.add_argument('--number_of_runs',help='the number of runs',default=5)

args = parser.parse_args()

data = np.load(args.data_location + '/data_small.npy')
labels = np.load(args.data_location + '/label_small.npy')

config = {'init':'random'}

perceptrons = []
for i in range(int(args.number_of_runs)):
    perceptrons.append(PLA(data.shape[-1],config))


for perceptron in perceptrons:
    for i in range(10000):
        if not perceptron.update(data, labels):
            break


y = np.linspace(0, 1, 10)
for perceptron in perceptrons:
    z = (1 * perceptron.weights[0] + y * perceptron.weights[1]) / - perceptron.weights[2]
    plt.plot(y, z, linewidth=2)
plt.scatter(data[:, 1], data[:, 2], c=labels)
plt.show()



