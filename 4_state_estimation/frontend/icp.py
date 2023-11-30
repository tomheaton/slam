# NOTE(JACOB): Reference https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb
# Basically a copy of the above

from curses import newpad
from re import X
import time
import timeit

from functools import partial
from sympy import Matrix, cos as s_cos, sin as s_sin
from math import sin, cos, atan2
import numpy as np

import matplotlib.pyplot as plt

import sys
import os
import colorutils
from copy import deepcopy


from example_data import ExampleDataGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'graph'))
from landmark import Landmark

# TODO(JACOB): Decide on a resonable threshold, taking errors into account
ZERO_ERROR_THRESHOLD = 0.0001
ITERATIONS = 20


class Plotter():
    @staticmethod
    def plot_data(p_values: list[list[Landmark]], q: list[Landmark], markersize_1=8, markersize_2=8):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.axis('equal')

        for i in range(len(p_values)):
            if i % 2 != 0:
                continue

            s = 1 if i == 0 or i == len(p_values) - 1 else 0.2

            p = p_values[i]
            x_p = []
            y_p = []

            for point in p:
                x_p.append(point.x)
                y_p.append(point.y)

            colour = colorutils.hsv_to_hex((i / len(p_values) * 200, s, 1))
            ax.plot(x_p, y_p, color=colour, markersize=markersize_1,
                    marker='o', linestyle=":", label=f"Step {i}")
        if q is not None:
            x_q = []
            y_q = []

            for point in q:
                x_q.append(point.x)
                y_q.append(point.y)
            ax.plot(x_q, y_q, color='blue', markersize=markersize_2,
                    marker='o', linestyle=":", label="True data")
        ax.legend()
        return ax

    @staticmethod
    def plot_values(values, label):
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.plot(values, label=label)
        ax.legend()
        ax.grid(True)
        # plt.show()


class MathsUtils():
    @staticmethod
    def __dR(theta):
        return np.array([[-sin(theta), -cos(theta)],
                        [cos(theta), -sin(theta)]])

    @staticmethod
    def R(theta):
        return np.array([[cos(theta), -sin(theta)],
                        [sin(theta), cos(theta)]])

    @staticmethod
    def jacobian(x, p_point):
        theta = x[2]
        J = np.zeros((2, 3))
        J[0:2, 0:2] = np.identity(2)
        J[0:2, [2]] = MathsUtils.__dR(theta).dot(p_point)
        return J

    @staticmethod
    def jacobian_0(x, p_point):
        J = np.zeros((2, 3))
        J[0:2, 0:2] = np.identity(2)
        J[0:2, [2]] = MathsUtils.__dR(0).dot(p_point)
        return J

    @staticmethod
    def error(x, p_point, q_point):
        rotation = MathsUtils.R(x[2])
        translation = x[0:2]
        prediction = rotation.dot(p_point) + translation
        return prediction - q_point

    @staticmethod
    def compute_normals(points, step=1):
        normals = [np.array([[0, 0]])]

        for i in range(step, len(points) - step):
            prev_point = points[i - step]
            next_point = points[i + step]
            dx = next_point.x - prev_point.x
            dy = next_point.y - prev_point.y
            normal = np.array([[0, 0], [-dy, dx]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal[[1], :])

        normals.append(np.array([[0, 0]]))
        return normals

    @staticmethod
    def get_rotation_matrix(angle):
        return Matrix([[s_cos(angle), -s_sin(angle)], [s_sin(angle), s_cos(angle)]])


class ICP():
    def __get_correspondence_indices(self, P: list[Landmark], Q: list[Landmark]):
        """For each point in P find closest one in Q."""
        p_size = len(P)
        q_size = len(Q)
        correspondences = []
        for i in range(p_size):
            p_point = P[i]
            min_dist = sys.maxsize
            chosen_idx = -1

            for j in range(q_size):
                q_point = Q[j]
                dist = np.linalg.norm(
                    q_point.as_vector() - p_point.as_vector())

                if dist < min_dist:
                    min_dist = dist
                    chosen_idx = j

            correspondences.append((i, chosen_idx))

        return correspondences

    def get_correspondence_ids(self, P: list[Landmark], Q: list[Landmark]):
        Q_normals = self.compute_normals(Q)

        # TODO: IDK difference between using theta or 0 in jacobian
        # Original source calculates theta, but never used it

        _, chi_values, corresp_indices = self.__icp_normal(
            P, Q, Q_normals, kernel=partial(self.__kernel, 3, 0.25), jacobian_0=True)

        # TODO: Test
        corresp_ids = []

        for pair in corresp_indices:
            corresp_ids.append((P[pair[0]].identifier, Q[pair[1]].identifier))

        # Return ids, final error
        return corresp_ids, chi_values[-1]

    def __prepare_system_normals(self, x, P: list[Landmark], Q: list[Landmark], correspondences, Q_normals, kernel=lambda distance: 1.0, jacobian_0=False):
        H = np.zeros((3, 3))
        g = np.zeros((3, 1))
        chi = 0
        for i, j in correspondences:
            p_point = P[i]
            q_point = Q[j]
            normal = Q_normals[j]
            e = normal.dot(self.error(
                x, p_point.as_vector(), q_point.as_vector()))
            weight = kernel(e)

            if jacobian_0:
                J = normal.dot(self.jacobian_0(x, p_point.as_vector()))
            else:
                J = normal.dot(self.jacobian(x, p_point.as_vector()))
            H += weight * J.T.dot(J)
            g += weight * J.T.dot(e)
            chi += e.T * e
        return H, g, chi

    def __icp_normal(self, P: list[Landmark], Q: list[Landmark], Q_normals, iterations=ITERATIONS, kernel=lambda distance: 1.0, jacobian_0=False):
        x = np.zeros((3, 1))
        chi_values = []
        x_values = [x.copy()]  # Initial value for transformation.
        #P = centre_p_on_q(P, Q)
        P_values = [deepcopy(P)]
        P_latest = deepcopy(P)
        corresp_values = []

        for _ in range(iterations):
            rot = MathsUtils.R(x[2])
            t = x[0:2]

            correspondences = self.__get_correspondence_indices(P_latest, Q)
            corresp_values.append(correspondences)

            H, g, chi = self.__prepare_system_normals(
                x, P, Q, correspondences, Q_normals, kernel, jacobian_0)

            dx = np.linalg.lstsq(H, -g, rcond=None)[0]
            x += dx

            x[2] = atan2(sin(x[2]), cos(x[2]))  # normalize angle

            chi_values.append(chi.item(0))  # add error to list of errors
            x_values.append(x.copy())

            rot = MathsUtils.R(x[2])
            t = x[0:2]

            # May need to copy P
            for i in range(len(P)):
                P_latest[i] = Landmark.from_vector(
                    rot.dot(P[i].as_vector()) + t)

            P_values.append(deepcopy(P_latest))

            # Skip Iterations if error is small enough
            # if chi_values[-1] < ZERO_ERROR_THRESHOLD:
            #    break

        corresp_values.append(corresp_values[-1])
        return P_values, chi_values, corresp_values

    def __kernel(self, threshold, sigma, error):
        # Sigmoid
        x = threshold - np.linalg.norm(error) * sigma
        value = 1 / (1 + np.exp(-x))

        return value

        if np.linalg.norm(error) < threshold:
            return 1.0
        return 0.0

    def debug_icp(self, P: list[Landmark], Q: list[Landmark], Q_normals, iterations=ITERATIONS, kernel=lambda distance: 1.0, jacobian_0=False):
        return self.__icp_normal(P, Q, Q_normals, iterations, kernel, jacobian_0)


def convert_vectors_to_landmarks(vectors):
    landmarks = []

    for i in range(vectors.shape[1]):
        landmarks.append(Landmark(vectors[0, i], vectors[1, i]))

    return landmarks


'''
def average(landmarks):
    avg_x = 0
    avg_y = 0

    for landmark in landmarks:
        avg_x += landmark.x
        avg_y += landmark.y

    return (avg_x / len(landmarks), avg_y / len(landmarks))

def centre_p_on_q(P, Q):
    avg_p = average(P)
    avg_q = average(Q)

    diff_x = avg_q[0] - avg_p[0]
    diff_y = avg_q[1] - avg_p[1]
    new_P = []
    for landmark in P:
        new_P.append(Landmark(landmark.x + diff_x, landmark.y + diff_y))

    return new_P
'''

##### GENERATE DATA #####
P, Q = ExampleDataGenerator.get_example_dataset_1(noise_sigma=0.00)
# TODO(JACOB): Get mean excluding outliers to get overall confidence score

P = convert_vectors_to_landmarks(P)
Q = convert_vectors_to_landmarks(Q)

# Test missing points
del P[2]
del P[14]
del P[14]
del P[14]

##### ICP #####
icp = ICP()

# Jacobian with 0 for theta
start = time.process_time()
P_values, chi_values, corresp_values = icp.debug_icp(P, Q)
print('Jacobian 0')
print(f'Took {time.process_time() - start}s')
print(f'Final error: {chi_values[-1]}')
Plotter.plot_values(chi_values, label="chi^2")
ax = Plotter.plot_data(P_values, Q, 4, 4)

print()

# Jacobian with actual theta
start = time.process_time()
P_values, chi_values, corresp_values = icp.debug_icp(P, Q)
print('Jacobian not 0')
print(f'Took {time.process_time() - start}s')
print(f'Final error: {chi_values[-1]}')
Plotter.plot_values(chi_values, label="chi^2")
ax = Plotter.plot_data(P_values, Q, 4, 4)

plt.show()
