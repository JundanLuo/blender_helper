import numpy as np


"""
    Cartesian (3D) Coordinate System:
        x-axis from screen to human eyes, y-axis points to the right, z-axis points to the top.
      z-axis (↑)
          |
          |
          |
          |______________ y-axis (→)
         /
        /
       /
      x-axis (out of the screen towards the viewer)
      
    We use a popular Cartesian system. Note it is different from the camera coordinate system we use.
"""


"""
    Spherical Coordinate System:
        Theta: angle from x-axis to the projection of the point on xy-plane
        Phi: angle from z-axis to the point
        Rho: distance from the origin to the point
"""


def camera_to_cartesian(cam_coordinate):
    """
        Camera space (3D) Coordinate System:
        x-axis points to the right, y-axis points to the top, z-axis from screen to human eyes.
        The z coordinate is negative for points in front of the camera.
      y-axis (↑)
          |
          |
          |
          |______________ x-axis (→)
         /
        /
       /
      z-axis (out of the screen towards the viewer)
    :param cam_coordinate:
    :return:
    """
    assert cam_coordinate.shape[-1] == 3, "The input should be in the shape of [..., 3]"
    x = cam_coordinate[..., 0:1]
    y = cam_coordinate[..., 1:2]
    z = cam_coordinate[..., 2:3]
    nx = z
    ny = x
    nz = y
    n = np.concatenate([nx, ny, nz], axis=-1)
    assert n.shape == cam_coordinate.shape, f"Shape mismatch: {n.shape} != {cam_coordinate.shape}"
    return n


def cart2sph(cart, eps=1e-6):
    """
        Cartesian-to-spherical transformation
        cart: ... X 3[xyz]
        ret:  ... X 3[ThetaPhiRho]
            theta range: [-pi, pi] in radians
            phi range: [0, pi] in radians
    """
    shape = cart.shape
    cart = cart.reshape(-1, 3)
    x = cart[:, 0:1]
    y = cart[:, 1:2]
    z = cart[:, 2:3]
    theta = np.arctan2(y, x)
    rho = np.linalg.norm(cart, ord=2, axis=-1, keepdims=True).clip(min=eps)
    phi = np.arccos(z / rho)
    sph = np.concatenate([theta, phi, rho], axis=-1)
    return sph.reshape(shape)


def sph2pix(sph, is_image_origin_top_left):
    """
        Spherical-to-pixel transformation (normalized coordinates)
        sph: L X 3[ThetaPhiRho]
        ret: L X 2[xy], range 0..1
    """
    theta, phi = sph[..., 0:1], sph[..., 1:2]
    theta = theta % (2 * np.pi)
    phi = phi % (2 * np.pi)
    assert np.all(theta >= 0) and np.all(theta <= 2 * np.pi), f"Theta out of range: {theta}"
    assert np.all(phi >= 0) and np.all(phi <= np.pi), f"Phi out of range: {phi}"
    x = (2 * np.pi - theta) / (2 * np.pi)
    y = phi / np.pi
    pix = np.concatenate([x, y], axis=-1)
    if not is_image_origin_top_left:
        # image origin is bottom left
        pix[..., 1] = 1 - pix[..., 1]
    assert np.all(pix >= 0) and np.all(pix <= 1), f"Pixel out of range: {pix}"
    return pix
