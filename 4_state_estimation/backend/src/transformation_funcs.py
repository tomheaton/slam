import numpy as np
# Debug
import math

def v2t(x):
    """
    Transforms (x,y,theta) vector into transformation matrix
    """
    ct = np.cos(x[2])
    st = np.sin(x[2])

    return np.array([
        [ct, -st, x[0]],
        [st,  ct, x[1]],
        [ 0,   0,   1],
    ])

def t2v(X):
    """
    Transforms transformation matrix into (x,y,theta) vector
    """
    result = np.array([X[0, 2], X[1, 2], np.arctan2(X[1,0], X[0,0])])
    # print 'X[0,0]: {}'.format(X[0,0])

    assert not math.isnan(X[0,0])

    # Debug
    # result = np.clip(result, 1000000, 0.00001)

    return result

def inv_t(X):
    """
    Computes and returns the inverse of a transformation matrix
    """

    R_T = X[0:2, 0:2].T
    t = X[0:2, 2]

    result = np.zeros((3,3))
    result[0:2, 0:2] = R_T
    result[0:2, 2] = np.dot(-R_T, t)
    result[2,2] = 1

    # Debug
    assert result.shape == (3,3)
    # result = np.clip(result, 1000000, 0.00001)

    return result

# Not used
# def inv_v(x):
    # """
    # Computes and returns the inverse of a (x,y,theta) vector
    # """
    # result = np.array([0, 0, -x[2]])

    # ct = np.cos(x[2])
    # st = np.sin(x[2])
    # # Rotation transposed
    # R_T = np.array([
        # [ ct, st],
        # [-st, ct]
    # ])
    # result[0:2] = -np.dot(R_T, x[0:2])

    # # Debug
    # assert result.shape == (3,)
    # # result = np.clip(result, 1000000, 0.00001)

    # return result
