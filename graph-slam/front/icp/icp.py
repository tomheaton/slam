import math


# TODO: use proper types / data structures.


def determine_correspondence(combined_set: set[any]) -> set[any]:
    """
    Determine the correspondence set.

    :param combined_set: a set.
    :return: a set.
    """

    return set()


def compute_transformation_params(correspondence_set: set[any]) -> tuple[any, any]:
    """
    Compute Transformation Parameters t and R.

    :param correspondence_set: a set.
    :return: a set.
    """

    transformation = 1
    rotation = 1

    return transformation, rotation


def error_function(transformation: any, rotation: any) -> int:
    """
    Error Function.

    :param transformation: Transformation.
    :param rotation: Rotation.
    :return: Error value.
    """

    return 1


threshold = 1

# Correspondence set.
C: set[any] = set()

xn = 1
yn = 1

xn_prime = xn

# Error values
e = math.inf
e_current = e


while e > e_current and e > threshold:
    c = determine_correspondence({yn, xn_prime})
    t, R = compute_transformation_params(C)
    xn_prime = R(xn - x0) + y0
    e = error_function(Rt * y0 - Rt * t, R)


print(f'{xn_prime = }')
