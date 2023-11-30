import numpy as np

from edge import Edge
from transformation_funcs import t2v, inv_t, v2t

from plot import plot_graph

def calculate_global_error(vertices, edges):
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

def calculate_err_and_jacobian(x1, x2, z):
    # Debug
    assert x1.shape == (3,)
    assert x2.shape == (3,)
    assert z.shape == (3,)

    zt_ij = v2t(z)
    vt_i = v2t(x1)
    vt_j = v2t(x2)

    # Debug
    assert zt_ij.shape == (3,3)
    assert vt_i.shape == (3,3)
    assert vt_j.shape == (3,3)

    R_i = vt_i[0:2, 0:2]
    R_ij= zt_ij[0:2, 0:2]
    ct = np.cos(x1[2])
    st = np.sin(x1[2])
    dRdt_i = np.array([[-st, ct],[-ct, -st]])

    err = t2v(np.dot(
        inv_t(zt_ij),
        np.dot(inv_t(vt_i), vt_j)
    ))

    A = np.zeros((3,3))
    A[0:2, 0:2] = -np.dot(R_ij.T, R_i.T)
    A[0:2, 2] = np.dot(
        R_ij.T,
        np.dot(dRdt_i, x2[0:2]-x1[0:2])
    )
    A[2,2] = -1

    B = np.zeros((3,3))
    B[0:2, 0:2] = np.dot(R_ij.T, R_i.T)
    B[2,2] = 1

    # Debug
    assert err.shape == (3,)
    assert A.shape == (3,3)
    assert B.shape == (3,3)

    return err, A, B


def get_graph_dx(vertices, edges):
    n = len(vertices)
    H = np.zeros((n,n))
    b = np.zeros((n,))

    needToFix = True

    for edge in edges:
        i = edge.idFrom*3
        j = edge.idTo*3
        x1 = vertices[i:i+3]
        x2 = vertices[j:j+3]

        e, A, B = calculate_err_and_jacobian(x1, x2, edge.estimate)

        b_i = -np.dot(e.T, np.dot(edge.info, A)).T
        b_j = -np.dot(e.T, np.dot(edge.info, B)).T
        H_ii = np.dot(A.T, np.dot(edge.info, A))
        H_ij = np.dot(A.T, np.dot(edge.info, B))
        H_jj = np.dot(B.T, np.dot(edge.info, B))

        # Debug
        assert b_i.shape == (3,)
        assert b_j.shape == (3,)
        assert H_ii.shape == (3,3)
        assert H_ij.shape == (3,3)
        assert H_jj.shape == (3,3)

        assert not any(np.isnan(b_i))
        assert not any(np.isnan(b_j))
        assert not np.isnan(H_ii).any()
        assert not np.isnan(H_ij).any()
        assert not np.isnan(H_jj).any()

        # TODO: add look that make sure there are no NaN values

        H[i:i+3, i:i+3] += H_ii
        H[i:i+3, j:j+3] += H_ij
        H[j:j+3, i:i+3] += H_ij
        H[j:j+3, j:j+3] += H_jj

        b[i:i+3] += b_i
        b[j:j+3] += b_j

        if (needToFix):
            H[i:i+3, i:i+3] = 1000*np.identity(3)
            needToFix = False

    dx = np.linalg.solve(H, -b)
    # dx = np.dot(np.linalg.inv(H), b)
    return dx


def optimise_graph(vertices, edges, max_iterations = 20, EPSILON = 0.001):
    print 'Starting optimisation...'

    for i in range(max_iterations):
        global_error = calculate_global_error(vertices, edges)
        print 'initial error: {}'.format(global_error)
        if global_error < EPSILON: # Converged
            print 'Convergence reached, graph optimised.'
            break

        dx = get_graph_dx(vertices, edges)

        # Debug
        assert dx.shape == vertices.shape

        vertices += dx

        # print 'plotting graph...'
        # plot_graph(vertices, edges)
