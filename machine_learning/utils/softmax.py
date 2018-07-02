# rudy 3/25/18
import numpy as np

def softmax(x): # compute sofmax of numpy array x
    '''
    scores = [1.0, 2.0, 3.0] should yield output
    [ 0.09003057  0.24472847  0.66524096]

    and 

    scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

    should yield np.array output
    [[ 0.09003057  0.00242826  0.01587624  0.33333333]
    [ 0.24472847  0.01794253  0.11731043  0.33333333]
    [ 0.66524096  0.97962921  0.86681333  0.33333333]]
    '''
    y=np.exp( x)
    return y/y.sum(axis=0)
    
