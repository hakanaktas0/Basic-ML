import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import PLA

parser = argparse.ArgumentParser()

parser.add_argument('--data_size',default='small',const='small',nargs='?',choices=['small','large'],help='specifies which dataset to use')
parser.add_argument('--initialization',help='specifies the initilization method of PLA',default='zero',const='zero',nargs='?',choices=['zero','random'])
parser.add_argument('--plot',help='if used, plots',action='store_true')
parser.add_argument('--savefig',help='if used, saves plot',action='store_true')
parser.add_argument('--visualization3D',help='if used, plots 3D otherwise 2D',action='store_true')
parser.add_argument('--data_location',help='please specify the path of the data (including PLA_data folder)',required=True)

args = parser.parse_args()



if args.data_size == 'small':
    data = np.load(args.data_location +   '/data_small.npy')
    labels = np.load( args.data_location + '/label_small.npy')
else:
    data = np.load(args.data_location + '/data_large.npy')
    labels = np.load(args.data_location + '/label_large.npy')


config = {}

config['plot'] = args.plot


config['init'] = args.initialization

config['3d'] = args.visualization3D

perceptron = PLA(data.shape[-1],config)

for i in range(10000):
    if not perceptron.update(data,labels):
        print('number of iterations to converge = ' + str(i))
        break


if args.plot:
    if args.visualization3D:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        x, y = np.meshgrid(x, y)
        z = (x * perceptron.weights[0] + y * perceptron.weights[1]) / - perceptron.weights[2]
        ax.plot_surface(x, y, z)
        if args.savefig:
            plt.savefig('figure-' + args.data_size +'-' + args.initialization + '-initialization-3D.png')
        plt.show()
    else:
        y = np.linspace(0, 1, 10)
        z = (1 * perceptron.weights[0] + y * perceptron.weights[1]) / - perceptron.weights[2]
        plt.scatter(data[:, 1], data[:, 2], c=labels)
        plt.plot(y, z, linewidth=2)
        if args.savefig:
            plt.savefig('figure-' + args.data_size +'-' + args.initialization + '-initialization-2D.png')
        plt.show()



