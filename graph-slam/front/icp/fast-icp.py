
def find_attitude_data(timestamp):
    """
    Find corresponding attitude data by time stamp.

    :param timestamp: The timestamp.
    :return: ?
    """

    pass


def find_displacement_data(timestamp):
    """
    Find corresponding displacement data by time stamp.

    :param timestamp: The timestamp.
    :return: ?
    """

    pass


def transform_cloud(a, b, c):
    """
    Transform the point cloud of previous frame into its own navigation coordinate system.

    :param a: Point Cloud.
    :param b: Point.
    :param c: Point difference.
    :return:
    """

    pass


def align_point_clouds(C, L, P, D):
    """
    This Program does stuff.

    :param C: Point cloud of current frame.
    :param L: Point cloud of previous frame.
    :param P: IMU data set.
    :param D: Encoder data set.
    :return: Point cloud of current frame that transformed to the global coordinate.
    """

    # TODO: initial registration.

    for p1 in P:
        print(f'{p1 = }')

        if L.timestamp == p1.timestamp:
            find_attitude_data(p1.timestamp)
            return p1

    for d1 in D:
        print(f'{d1 = }')

        if L.timestamp == d1.timestamp:
            find_displacement_data(d1.timestamp)
            return d1

    transform_cloud(L, p1, 0)

    for p2 in P:
        print(f'{p2 = }')

        if C.timestamp == p2.timestamp:
            find_attitude_data(p2.timestamp)
            return p2

    for d2 in D:
        print(f'{d2 = }')

        if C.timestamp == d2.timestamp:
            find_displacement_data(d2.timestamp)
            return d2

    transform_cloud(C, p2, d2 - d1)

    return C
