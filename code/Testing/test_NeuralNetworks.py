import unittest
import numpy as np
import sys
# to import from a parent directory
sys.path.append('../')
from utils import mape
from NeuralNetworks.FeedForward import FeedForward
from NeuralNetworks.lstm import LSTM

class NeuralNetworksTestCase(unittest.TestCase):
    """ test to ensure each neural network can predict
        a straight line with high accuracy """

    def test_LSTM(self):
        # create recurrent neural network
        NN = LSTM()

        # create training and testing inputs and targets
        train_input_1 = [[100, 100] for i in range(100)]
        train_input_2 = train_input_1
        train_input_3 = train_input_1
        train_target = [[100] for i in range(100)]

        test_input_1 = [[101, 101] for i in range(50)]
        test_input_2 = test_input_1
        test_input_3 = test_input_1
        test_target = [[101] for i in range(50)]

        # normalize and convert to arrays
        train_input_1 = np.array(train_input_1) / 1000
        train_input_2 = train_input_1
        train_input_3 = train_input_1
        train_target = np.array(train_target) / 1000

        test_input_1 = np.array(test_input_1) / 1000
        test_input_2 = test_input_1
        test_input_3 = test_input_1       
        test_target = np.array(test_target) / 1000

        # number of training cycles
        epochs = 100

        # train the neural network
        for e in range(epochs):
            for p in train_input_1:
                train_output = NN.train(train_input_1, train_input_2, train_input_3, train_target)

        # test on unseen data
        test_output = NN.test(test_input_1, test_input_2, test_input_3)

        # de-normalize
        train_output *= 1000
        train_target *= 1000

        test_output *= 1000
        test_target *= 1000

        self.assertGreaterEqual(100 - mape(train_target, train_output), 99.00)
        self.assertGreaterEqual(100 - mape(test_target, test_output), 97.00)


if __name__ == '__main__':
    unittest.main()