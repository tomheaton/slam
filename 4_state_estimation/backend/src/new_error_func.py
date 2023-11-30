import numpy as np
from transformation_funcs import t2v, v2t, inv_t


# OLD
def calculate_global_error(vertices, edges):
    print("edges: ", edges)
    total_error = np.float64(0.0)
    for edge in edges:
        x1 = vertices[edge.idFrom*3:edge.idFrom*3+3]
        x2 = vertices[edge.idTo*3:edge.idTo*3+3]
        z12 = edge.estimate
        info = edge.info

        e = t2v(
            np.dot(
                inv_t(v2t(z12)),
                np.dot(inv_t(v2t(x1)), v2t(x2))
            )
        )

        # Debug
        assert e.shape == (3,)

        err = np.dot(e.T, np.dot(info, e))

        if err > 0.00001: # This because im getting NaN errors
            total_error = total_error + err

    return total_error

# NEW
def calc_error_new(vertices, edges):

    total_error = np.float64(0.0)

    for edge in edges:
        x1 = vertices[edge.idFrom*3:edge.idFrom*3+3] # the first pose
        x2 = vertices[edge.idTo*3:edge.idTo*3+3] # the second pose
        print(x1)
        print(x2)
        z12 = edge.estimate # the measurement
        info = edge.info # the information matrix for the estimate

    return total_error
