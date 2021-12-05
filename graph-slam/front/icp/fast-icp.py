global_transform = 0


# TODO: add types / proper data structures to variables


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
    :return: ?
    """

    pass


def pair_align(a, b):
    """
    Accurate Registration.

    :param a: Point cloud for previous frame.
    :param b: Point cloud for current frame.
    :return: The matrix.
    """

    x = 0

    return x


def transform_point_cloud(a, b):
    """
    Transform the point cloud to the global coordinate system which is the
        navigation coordinate system of the first input point cloud.

    :param a: Point cloud for current frame.
    :param b: New transform.
    :return: ?
    """

    pass


def compute_transform(a, b):
    """
    Calculate the transformation from C1 to C2, C1 is the point cloud obtained by transforming C
        into its own navigation coordinate system, C2 is the point cloud obtained by
        transforming c into the navigation coordinate system of previous frame.

    :param a: Point cloud for current frame.
    :param b: New transform.
    :return: ?
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

    global global_transform

    # Initial Registration:

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

    # Accurate Registration:

    pair_transform = pair_align(L, C)

    # Transform the point cloud to the global coordinate system which is the navigation
    # coordinate system of the first input point cloud.
    transform_point_cloud(C, global_transform * pair_transform)

    _transform = compute_transform(p2, d2 - d1)

    # Update global transformation.
    global_transform = global_transform * pair_transform * _transform

    return C
