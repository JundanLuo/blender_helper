import mathutils


def align_direction(initial_direction, direction, return_euler=True):
    """
    Aligns the initial direction to the direction. Return the rotation.
    :param initial_direction:
    :param direction:
    :param return_euler:
    :return: Euler angles in radians, or a Quaternion if return_euler=False.
    """
    initial_direction = mathutils.Vector(initial_direction)
    initial_direction.normalize()
    direction = mathutils.Vector(direction)
    direction.normalize()
    rotation = initial_direction.rotation_difference(direction)
    if return_euler:
        rotation = rotation.to_euler()
    return rotation