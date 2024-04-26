import numpy as np
import argparse
from model import PLA



parser = argparse.ArgumentParser()

parser.add_argument('--data_location',help='please specify the path of the data (including PLA_data folder)',required=True)
parser.add_argument('--number_of_runs',help='the number of runs',default=100,nargs='?',const=100)

args = parser.parse_args()




small_data = np.load(args.data_location +   '/data_small.npy')
small_labels = np.load( args.data_location + '/label_small.npy')

large_data = np.load(args.data_location + '/data_large.npy')
large_labels = np.load(args.data_location + '/label_large.npy')


config = {}



config['init'] = 'random'


number_of_iter_small = np.zeros(args.number_of_runs)
number_of_iter_large = np.zeros(args.number_of_runs)

for data_size in ['small','large']:
    if data_size == 'small':
        data = small_data
        labels = small_labels
    else:
        data = large_data
        labels = large_labels
    for a in range(args.number_of_runs):
        perceptron = PLA(data.shape[-1],config)
        for i in range(10000):
            if not perceptron.update(data,labels):
                if data_size == 'small':
                    number_of_iter_small[a] = i
                else:
                    number_of_iter_large[a] = i
                break

print('Mean number of iter for small dataset : ' + str(np.mean(number_of_iter_small)) + '   varience : '  + str(np.var(number_of_iter_small)))
print()
print('Mean number of iter for large dataset : ' + str(np.mean(number_of_iter_large)) + '   varience : '  + str(np.var(number_of_iter_large)))