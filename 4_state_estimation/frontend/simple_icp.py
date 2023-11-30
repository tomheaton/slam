import math
import random

import matplotlib.pyplot as plt
import numpy as np

# NOTE(JACOB): Try stochastic?
# NOTE(JACOB): Might need to have a certainty value with a cut-off to avoid wrong predictions

# Algorithm (source: https://en.wikipedia.org/wiki/Iterative_closest_point):
# 1. For each point in source point cloud, match closest point in reference point cloud
# 2. Estimate combination of rotation and translation using root mean square point to point distance. This may involve weighting points & rejecting outliers prior to alignment
#       NOTE(JACOB): maybe less weighting on further away cones?
# 3. Transform source points
# 4. Iterate

ALIGN_INTERVAL = 1
MAX_ITERATIONS = 100

ZERO_ERROR_THRESHOLD = 0.001


def get_closest_point(reference: list[tuple[int, float, float]], point: tuple[int, float, float]) -> tuple[int, float]:
    """
    :param reference: The graph to find the closest point in. A list of (id, x, y) tuples for each landmark
    :param point: The point (id, x, y) to compare to the reference

    :return: A tuple with the id of the reference point and the square distance between the points
    """

    min_point = None
    min_dist_sqr = math.inf

    for ref_point in reference:
        # Pythagoras
        x_diff = ref_point[1] - point[1]
        y_diff = ref_point[2] - point[2]
        dist_sqr = (x_diff * x_diff) + (y_diff * y_diff)

        if dist_sqr < min_dist_sqr:
            min_point = ref_point[0]
            min_dist_sqr = dist_sqr

    return (min_point, min_dist_sqr)

# NOTE(JACOB): Must ensure more reference points than source points or find another way to deal with this


def compute_error(reference: list[tuple[int, float, float]], source: list[tuple[int, float, float]]) -> float:
    """
    :param reference: The graph to match the source to. A list of (id, x, y) tuples for each landmark
    :param source: A set of points to match to the reference. A list of (id, x, y) tuples for each landmark

    :return: A value for the error between each landmark using root mean square
    """

    if len(reference) < len(source):
        raise ValueError(
            "There must be more reference points than source points")

    error = 0

    # Check each source point's distance to the closest reference point
    # Do this since there might not be as many source points as there are reference points
    for source_point in source:
        _, dist_sqr = get_closest_point(reference, source_point)
        error += dist_sqr

    return math.sqrt(error / len(source))  # Root mean square value


def get_centre_of_mass(points: list[tuple[int, float, float]]) -> tuple[float, float]:
    """
    :param points: The points to find the centre of mass of

    :return: The (x, y) location of the centre of mass as a tuple
    """

    # TODO(JACOB): Use weighting here for each point?

    centre_x = 0
    centre_y = 0

    for point in points:
        centre_x += point[1]
        centre_y += point[2]

    return (centre_x / len(points), centre_y / len(points))


def rotate_points(points: list[tuple[int, float, float]], pivot: tuple[float, float], delta_theta: float) -> list[tuple[int, float, float]]:
    """
    :param points: The points to rotate
    :param pivot: The (x, y) coordinate to pivot around
    :param delta_theta: The angle, in radians, to rotate

    :return: A new source list with all the points rotated around the pivot by 'angle' degrees
    """

    new_points = []

    for point in points:
        adjusted_x = point[1] - pivot[0]
        adjusted_y = point[2] - pivot[1]

        # Calculate angle, going anticlockwise, from +ve x axis
        theta = math.atan2(adjusted_y, adjusted_x)
        new_theta = theta + delta_theta

        # TODO(JACOB): Optimise? Square root slow?
        r = math.sqrt(adjusted_x*adjusted_x + adjusted_y*adjusted_y)

        new_x = r * math.cos(new_theta)
        new_y = r * math.sin(new_theta)

        new_points.append((point[0], new_x + pivot[0], new_y + pivot[1]))

    return new_points


# NOTE(JACOB): Must be similar section of reference. Whole graph for reference will not work as this will move the source to the wrong location
def align_translation_com(reference: list[tuple[int, float, float]], source: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """
    :param reference: The graph to match the source to. A list of (id, x, y) tuples for each landmark
    :param source: A set of points to match to the reference. A list of (id, x, y) tuples for each landmark

    :return: A new source list with all the points shifted to align the 'centre of mass' with the reference
    """

    reference_com = get_centre_of_mass(reference)
    source_com = get_centre_of_mass(source)

    diff_x = source_com[0] - reference_com[0]
    diff_y = source_com[1] - reference_com[1]

    new_source = []
    for point in source:
        new_source.append((point[0], point[1] - diff_x, point[2] - diff_y))

    return new_source


def align_rotation(reference: list[tuple[int, float, float]], source: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """
    :param reference: The graph to match the source to. A list of (id, x, y) tuples for each landmark
    :param source: A set of points to match to the reference. A list of (id, x, y) tuples for each landmark

    :return: A new source list with all the points rotated to align best with the reference
    """

    # Rotate around centre of mass of source
    source_com = get_centre_of_mass(source)

    best_new_source = None
    best_error = math.inf

    for theta in range(0, 360, ALIGN_INTERVAL):
        new_source = rotate_points(source, source_com, theta / 180 * math.pi)
        error = compute_error(reference, new_source)

        if error < best_error:
            best_error = error
            best_new_source = new_source

    # TODO(JACOB): Maybe return error to use for certainty of prediction
    return best_new_source, best_error


def compute_icp(reference: list[tuple[int, float, float]], source: list[tuple[int, float, float]]) -> dict[int, int]:
    """
    :param reference: The graph to match the source to. A list of (id, x, y) tuples for each landmark
    :param source: A set of points to match to the reference. A list of (id, x, y) tuples for each landmark

    :return: A dictionary with the key of the source point and value of the reference point
    """

    # Simple algorithm to test concept:
    # Calculate 'centre of mass' of both points and align them
    # Then loop from angles 0-360 (with interval) and find one with smallest error

    prev_error = 0
    checking_zero_error = False
    errors = [compute_error(reference, source)]
    for i in range(MAX_ITERATIONS):
        # TODO(JACOB): Do not need to translate again - centre of mass will not change
        # TODO(JACOB): Match closest points and calculate average transformation
        source = align_translation_com(reference, source)
        source, error = align_rotation(reference, source)

        if abs(error - errors[-1]) < ZERO_ERROR_THRESHOLD:
            print(error, prev_error)
            # TODO(JACOB): This could still get stuck at a local min - this is where stochastic could help
            if checking_zero_error:
                break
            else:
                checking_zero_error = True
        else:
            checking_zero_error = False

        errors.append(error)
        prev_error = error

    global axis
    print(errors)
    axis[1].plot(np.arange(len(errors)), errors)
    plot_points(source, axis[0])

    # Match points up
    associations = {}
    for point in source:
        closest_id, _ = get_closest_point(reference, point)
        associations[point[0]] = closest_id

    return associations


def add_noise(points, mag):
    new_points = []

    for point in points:
        new_points.append(
            (point[0], point[1] + random.gauss(0, mag), point[2] + random.gauss(0, mag)))

    return new_points


def plot_points(points, axis):
    points = np.asarray(points)
    axis.scatter(points[:, 1], points[:, 2])
    axis.plot(points[:, 1], points[:, 2])


figure, axis = plt.subplots(1, 2)

reference_test_data = [(0, 13, 112), (1, 15, 89), (2, 20, 68), (3, 32, 55), (4, 46, 60), (5, 59, 75), (6, 73, 82), (6, 83, 81), (7, 88, 72), (
    8, 84, 61), (9, 71, 52), (10, 67, 41), (11, 76, 35), (12, 93, 36), (13, 105, 48), (14, 106, 62), (15, 106, 76), (16, 101, 91), (17, 98, 101)]
source_test_data = [(100, 68, 111), (101, 46, 102), (102, 28, 91), (103, 19, 76), (104, 28, 64), (105, 46, 56), (106, 57, 45), (106, 59, 35), (107, 52, 28),
                    (108, 40, 28), (109, 27, 38), (110, 16, 39), (111, 14, 28), (112, 19, 12), (113, 34, 4), (114, 47, 7), (115, 61, 12), (116, 74, 21), (117, 82, 27)]

source_test_data = add_noise(source_test_data, 50)

print(compute_icp(reference_test_data, source_test_data))
#plot_points(source_test_data, axis[0])
plot_points(reference_test_data, axis[0])

plt.show()
