import tensorflow as tf
import math
import numpy as np
import unittest
# from ddt import ddt, data, unpack
from StackedDAE import *

class TestStackedDAE(unittest.TestCase):
    def test_stackedDAE(self):
        # generate the stacked autoEncoder model
        myops = StackedDAE(tf.placeholder(tf.float32,[None,8]),[8,4,2],is_training = True)
        # script to generate noised data
        def generate_test_data(columns, rows, noise_lvl = 0.25):
            data = np.zeros([1,columns])
            for _ in range(math.ceil(rows/columns)):
                actual = np.eye(columns)
                noise = (np.random.random([columns,columns])-.5)*noise_lvl
                data = np.vstack([data,actual+noise])
            return data[1:rows+1]
        # get training data
        data = generate_test_data(8, 2**8)
        myops.train(data=data)
        # get testing data pairs
        myops = StackedDAE(tf.placeholder(tf.float32,[None,8]),[8,4,2],is_training = False)
        d1 = generate_test_data(8,8)
        d2 = generate_test_data(8,8)
        # get encoded data
        d1, d2 = myops.encode(d1), myops.encode(d2)
        # calculate distance between vectors
        def distance(v1,v2):
            return sum((v1-v2)*(v1-v2))
        # calculate pairwise distance between encoded noisy vectors
        inner_distances = [distance(v1,v2) for i,v1 in enumerate(d1) for j,v2 in enumerate(d2) if i == j]
        outer_distances = [distance(v1,v2) for i,v1 in enumerate(d1) for j,v2 in enumerate(d2) if i != j]
        self.assertEqual(True, np.max(inner_distances) < np.max(outer_distances))
        self.assertEqual(True, np.min(inner_distances) < np.min(outer_distances))
        self.assertEqual(True, np.mean(inner_distances) < np.mean(outer_distances))


if __name__ == '__main__':
    unittest.main(verbosity=1)

