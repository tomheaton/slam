import random

from math import pi, sin, cos
import numpy as np


class ExampleDataGenerator():
    def __get_test_data():
        # initialize pertrubation rotation
        angle = random.random() * pi - (pi / 2)
        R_true = np.array([[cos(angle), -sin(angle)],
                          [sin(angle), cos(angle)]])
        t_true = np.array([[-2], [5]])

        # Generate data as a list of 2d points
        num_points = 30
        true_data = np.zeros((2, num_points))
        true_data[0, :] = range(0, num_points)
        true_data[1, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :])

        # Move the data
        moved_data = R_true.dot(true_data) + t_true

        return moved_data, true_data

    def __get_test_data_2():
        # initialize pertrubation rotation
        angle = random.random() * pi - (pi / 2)
        R_true = np.array([[cos(angle), -sin(angle)],
                          [sin(angle), cos(angle)]])
        t_true = np.array([[-2], [5]])

        # Generate data as a list of 2d points
        num_points = 30
        true_data = np.zeros((2, num_points))
        # (0.2t*sin4t, 0.2t^{2}*cos t)
        min_t = 0.5
        max_t = 1.3 * pi
        t = np.arange(num_points) / num_points * (max_t - min_t) + min_t
        true_data[0, :] = t * np.sin(4 * t)
        true_data[1, :] = t ** 2 * np.cos(t)

        # Move the data
        moved_data = R_true.dot(true_data) + t_true

        return moved_data, true_data

    def __add_outliers(P, num, sigma):
        P_outliers = P.copy()

        for _ in range(num):
            index = random.randrange(P_outliers.shape[1])
            P_outliers[:, index] += np.array(
                [random.gauss(0, sigma), random.gauss(0, sigma)])

        #center_of_P_outliers = np.array([P_outliers.mean(axis=1)]).T
        #center_of_Q = np.array([Q.mean(axis=1)]).T
        #P_centered_outliers = P_outliers - center_of_P_outliers
        #Q_centered = Q - center_of_Q

        return P_outliers

    def __add_noise(P, sigma):
        P_outliers = P.copy()

        for i in range(P.shape[1]):
            P_outliers[:, i] += np.array(
                [random.gauss(0, sigma), random.gauss(0, sigma)])

        #center_of_P_outliers = np.array([P_outliers.mean(axis=1)]).T
        #center_of_Q = np.array([Q.mean(axis=1)]).T
        #P_centered_outliers = P_outliers - center_of_P_outliers
        #Q_centered = Q - center_of_Q

        return P_outliers

    def __add_noise_and_outliers(P, noise_sigma, num_outliers, outliers_sigma):
        new_P = ExampleDataGenerator.__add_noise(P, noise_sigma)
        new_P = ExampleDataGenerator.__add_outliers(
            new_P, num_outliers, outliers_sigma)

        return new_P

    # param(noise): sigma for gaussian noise
    # param(outliers): tuple - (num outliers, sigma for gaussian)
    def get_example_dataset_1(noise_sigma=0, num_outliers=0, outliers_sigma=0):
        P, Q = ExampleDataGenerator.__get_test_data()
        P = ExampleDataGenerator.__add_noise_and_outliers(
            P, noise_sigma, num_outliers, outliers_sigma)

        return P, Q

    # param(noise): sigma for gaussian noise
    # param(outliers): tuple - (num outliers, sigma for gaussian)
    def get_example_dataset_2(noise_sigma=0, num_outliers=0, outliers_sigma=0):
        P, Q = ExampleDataGenerator.__get_test_data_2()
        P = ExampleDataGenerator.__add_noise_and_outliers(
            P, noise_sigma, num_outliers, outliers_sigma)

        return P, Q
