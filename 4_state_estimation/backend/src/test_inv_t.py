import sys
import math
import numpy as np

from transformation_funcs import inv_t


def test(amount):

    count = 0

    for _ in range(0, amount):

        # matrix to test
        # m = np.random.randint(100, size=(3, 3))
        m = np.random.rand(3, 3)

        # inv_t() inverse matrix
        inverse_a = inv_t(m)

        # np.linalg.inv() inverse matrix
        inverse_b = np.linalg.inv(m)

        # count += 1 if np.array_equal(inverse_a, inverse_b) else 0
        count += 1 if np.allclose(inverse_a, inverse_b) else 0

        print(inverse_a)
        print("\n")
        print(inverse_b)

    print("inverse matrix testing with {}/{} correct.".format(count, amount))


if __name__ == '__main__':
    print("testing...");
    # test(1)
    xin = []
    output = []

    with open('./inv_t/input_test.txt', 'r') as f:
        lines = f.readlines()
        for i in range(1000):
            x = np.array([a.split() for a in lines[i*4:i*4+3]], dtype=np.float64)
            xin.append(x)

    with open('./inv_t/result_test.txt', 'r') as f:
        lines = f.readlines()
        for i in range(1000):
            x = np.array([a.split() for a in lines[i*3:i*3+3]], dtype=np.float64)
            output.append(x)

    # TESTING
    for i, x in enumerate(xin):
        if i > 1000:
            break
        result = np.around(inv_t(x), decimals=6)
        a = np.around(output[i], decimals=6)
        eq = np.allclose(a, result, rtol=1e-04, atol=1e-05)
        if not eq:
            print('ERROR: iter#:{}').format(i)
            print('calculated:')
            print(result)
            print('matlab output:')
            print(a)
            # if input('q to exit: ') == 'q':
            #     sys.exit()
        # else:
        #     print("TRUE")
    print('done')
