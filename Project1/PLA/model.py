import numpy as np

class PLA:
    def __init__(self, num_of_weights, config):
        self.config = config
        if config['init'] == 'zero':
            self.weights = np.zeros(num_of_weights)
        elif config['init'] == 'random':
            self.weights = np.random.random(num_of_weights)

    def evaluate(self,x):
        if np.dot(x,self.weights) > 0:
            return 1
        else:
            return -1

    def update(self,xs,ys):
        for x,y in zip(xs,ys):
            if y != self.evaluate(x):
                self.weights += x*y
                return True
        return False