from scipy.spatial.transform import Rotation
import scipy
import numpy as np
import math


def vector(x, y, z):
    return np.array([x, y, z], dtype=np.float32)


def rotation_xyz(x, y, z):
    """Create an XYZ Euler rotation from angles in degrees."""
    return Rotation.from_euler('xyz', [x, y, z], degrees=True)


class Pose(object):
    loc: np.ndarray  # np.array, shape(3,)
    orient: Rotation  # scipy.spatial.transform.rotation

    def __init__(self, location: np.ndarray = np.array([0, 0, 0]),
                 orientation: Rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)):
        """
        :param location: 3D location
        :param orientation: Rotation or euler angles (xyz) in degrees
        """
        if isinstance(location, list) or isinstance(location, tuple):
            location = np.array(location, dtype=np.float32)
        self.loc = location.flatten().astype(np.float32)
        assert location.shape == (3,), f"Error shape: {location.shape}"
        if isinstance(orientation, list) or isinstance(orientation, tuple) or isinstance(orientation, np.ndarray):
            assert len(orientation) == 3, f"Error length: {len(orientation)}"
            orientation = Rotation.from_euler('xyz', orientation, degrees=True)
        self.orient = orientation

    def transform(self, R, T):
        l = R.apply(self.loc) + T
        o = R * self.orient
        return Pose(l, o)

    def __str__(self):
        return f"location: {self.loc};    orientation: {self.orient.as_euler('xyz', degrees=True)}"


zero_pose = Pose(location=np.array([0, 0, 0]),
                 orientation=Rotation.from_euler('xyz', [0, 0, 0], degrees=True))


class Ray(object):
    origin = None
    direct = None

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin.squeeze().astype(np.float32)
        self.direct = direction.squeeze().astype(np.float32)
        assert self.origin.shape == self.direct.shape == (3, )

    def __str__(self) -> str:
        return f"Origin: {self.origin},  Direction: {self.direct}"


class Plane(object):
    DEFAULT_NORMAL = vector(0, 0, 1)
    loc: np.ndarray
    normal: np.ndarray
    d: float
    orient: Rotation

    def __init__(self, location: np.ndarray, normal: np.ndarray):
        self.loc = location.flatten().astype(np.float32)
        self.normal = normal.flatten().astype(np.float32)
        assert self.loc.shape == self.normal.shape == (3, )
        self.d = -self.loc.dot(self.normal)
        self.orient, _ = Rotation.align_vectors(self.normal.reshape(1, 3),
                                             self.DEFAULT_NORMAL.reshape(1, 3))

    def intersect(self, ray: Ray):
        if abs(ray.direct.dot(self.normal)) < 1e-5:  # parallel
            return None
        t = (self.loc - ray.origin).dot(self.normal) / ray.direct.dot(self.normal)
        if t < 0:  # no intersection
            return None
        return ray.origin + t * ray.direct

    def oblique_projection_from(self, points_3d: np.ndarray, direction: np.ndarray):
        points_3d = points_3d.reshape(3, -1)
        src_normal = direction.reshape(3, 1)
        tg_normal = self.normal.reshape(3, 1)
        nn = (src_normal @ tg_normal.transpose()) / (src_normal.transpose() @ tg_normal)
        nn = np.identity(3) - nn
        o = self.loc.reshape(3, -1)
        return o + nn@(points_3d-o)


class ImagePlane(object):
    '''
        plane.origin is the 3D location of the left-top corner
    '''
    plane = None
    x_direct = None
    y_direct = None
    width = None
    height = None
    m_plane2d_2_space3d = None

    def __init__(self, origin: np.ndarray, normal: np.ndarray,
                 x_direct: np.ndarray, y_direct: np.ndarray,
                 width, height):
        # Input
        self.plane = Plane(origin, normal)
        self.x_direct = x_direct.squeeze().astype(np.float32)
        self.y_direct = y_direct.squeeze().astype(np.float32)
        assert x_direct.shape == y_direct.shape == (3, )
        self.width = width
        self.height = height
        # Derived
        self.m_plane2d_2_space3d = np.column_stack((self.width*self.x_direct,
                                                    self.height*self.y_direct)).astype(np.float32)

    def get_3D_location_on_image_at(self, xy: np.ndarray):
        points_2d = xy.reshape(2, -1)
        return self.plane.loc.reshape(3, 1) + self.m_plane2d_2_space3d @ points_2d


class OrthographicCamera(object):
    """
    Orthographic Camera class.

    Camera space (3D) Coordinate System:
        x-axis points to the right, y-axis points to the top, z-axis from screen to human eyes.
        Thus, z coordinate is negative.
      y-axis (↑)
          |
          |
          |
          |______________ x-axis (→)
         /
        /
       /
      z-axis (out of the screen towards the viewer)

    Image space (2D) Coordinate System:
        x-axis points to the right, y-axis points to the top. The origin for the pixel (2D) is at the bottom-left corner.
        y-axis (↑)
            |
            |
            |
            |______________ x-axis (→)
         (0, 0)
        Origin (bottom-left corner)

    Note: This is different from the image matrix (numpy or OpenCV), where the origin is at the top-left corner.

    """

    DEFAULT_LOOKAT = vector(0, 0, -1)

    p = None
    # intrinsic
    fx, fy = None, None
    img_W, img_H = None, None
    _intrinsic_matrix = None  # 3 X 3
    # extrinsic
    R: Rotation
    T: np.ndarray
    _extrinsic_matrix = None  # 3 X 4
    # projection matrix
    _orthographic_projection_matrix = None  # 3 X 4
    projection_matrix = None  # 3 X 4

    def __init__(self, pose: Pose, img_size, fx, fy):
        # extrinsic
        self.p = pose
        self.R = self.p.orient.inv()
        self.T = - self.R.apply(self.p.loc)
        self._extrinsic_matrix = np.concatenate([self.R.as_matrix(), self.T.reshape(3, 1)], axis=1)
        # intrinsic
        self.img_W, self.img_H = img_size
        self.fx, self.fy = fx, fy
        self._intrinsic_matrix = np.array([
            [fx, 0, -self.img_W / 2.0],
            [0, fy, -self.img_H / 2.0],
            [0, 0, -1],
        ], dtype=np.float32)
        # projection
        self._orthographic_projection_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1],
        ], dtype=np.float32)
        self.projection_matrix = self._orthographic_projection_matrix @ self.extrinsic_4x4()

    def transform(self, R, T):
        """Transform the camera by R and T"""
        return OrthographicCamera(self.p.transform(R, T), (self.img_W, self.img_H), self.fx, self.fy)


    def get_look_at(self):
        """Return the look-at direction of the camera in world space"""
        return self.p.orient.apply(self.DEFAULT_LOOKAT)

    def extrinsic_3x4(self):
        """Return the 3x4 extrinsic matrix of the camera"""
        return self._extrinsic_matrix

    def extrinsic_4x4(self):
        """Return the 4x4 extrinsic matrix of the camera"""
        return np.concatenate([self._extrinsic_matrix, np.array([[0, 0, 0, 1]])], axis=0)

    def intrinsic_3x3(self):
        """Return the 3x3 intrinsic matrix of the camera"""
        return self._intrinsic_matrix

    def intrinsic_3x4(self):
        """Return the 3x4 intrinsic matrix of the camera"""
        return np.concatenate([self._intrinsic_matrix, np.zeros((3, 1))], axis=1)

    def projection_3x4(self):
        """Return the 3x4 projection matrix of the camera.
        Projection matrix = orthographic_projection_matrix @ extrinsic_matrix
        """
        return self.projection_matrix


class Camera(object):
    """
    Pin-hole camera class.

    Camera space (3D) Coordinate System:
        x-axis points to the right, y-axis points to the top, z-axis from screen to human eyes.
        Thus, z coordinate is negative.
      y-axis (↑)
          |
          |
          |
          |______________ x-axis (→)
         /
        /
       /
      z-axis (out of the screen towards the viewer)

    Image space (2D) Coordinate System:
        x-axis points to the right, y-axis points to the top. The origin for the pixel (2D) is at the bottom-left corner.
        y-axis (↑)
            |
            |
            |
            |______________ x-axis (→)
         (0, 0)
        Origin (bottom-left corner)

    Note: This is different from the image matrix (numpy or OpenCV), where the origin is at the top-left corner.

    """
    DEFAULT_LOOKAT = vector(0, 0, -1)

    p = None
    # intrinsic
    img_W, img_H = None, None
    fx, fy = None, None
    _intrinsic_matrix = None
    # extrinsic
    R = None
    T = None
    _extrinsic_matrix = None

    def __init__(self, pose: Pose, img_size, FOV_deg) -> None:
        """
        :param pose: Pose
        :param img_size: (width, height)
        :param FOV_deg: Fov_x in degree
        """
        # extrinsic
        self.p = pose
        self.R = self.p.orient.inv()
        self.T = - self.R.apply(self.p.loc)
        self._extrinsic_matrix = np.concatenate([self.R.as_matrix(), self.T.reshape(3, 1)], axis=1)
        # intrinsic
        assert len(img_size) == 2, f"Error shape: {img_size}"
        self.img_W, self.img_H = img_size
        self.FOV_deg = FOV_deg
        fx = fy = 0.5 * self.img_W / math.tan(np.deg2rad(FOV_deg / 2.0))
        self._intrinsic_matrix = np.array([
            [fx, 0, -self.img_W / 2.0],
            [0, fy, -self.img_H / 2.0],
            [0, 0, -1],
        ]).astype(np.float32)
        self.fx, self.fy = fx, fy

    def camera_view(self, pose_or_points):
        """Return the pose_or_points in camera space"""
        if isinstance(pose_or_points, Pose):
            world_pose = pose_or_points
            return world_pose.transform(self.R, self.T)
        elif isinstance(pose_or_points, np.ndarray):
            points = pose_or_points
            assert points.shape[-1] == 3, f"Error shape: {points.shape}"
            shape = points.shape
            v = self.R.apply(points.reshape(-1, 3)) + self.T.reshape(1, 3)
            v = v.reshape(shape)
            return v
        else:
            raise ValueError(f"Invalid input type: {type(pose_or_points)}")

    def camera_image(self, v: np.ndarray, is_in_world_space,  is_2D_origin_bottom_left):
        """Return the image coordinates of the 3D points after projection.
        v: 3D points, shape(..., 3)
        """
        v_shape = list(v.shape)
        if is_in_world_space:
            # transform the points to camera space
            v = self.camera_view(v)
        assert v.shape[-1] == 3, f"Error shape: {v.shape}"
        v = v.reshape(-1, 3)  # N X 3
        v = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)  # N X 4
        assert v.shape[-1] == 4, f"Error shape: {v.shape}"
        v = v.transpose()  # 4 X N
        v_image = self.intrinsic_3x4() @ v  # 3 X N
        v_image[:, :] /= v_image[-1:, :]  # 3 X N
        v_image = v_image[:-1, :]  # 2 X N
        if not is_2D_origin_bottom_left:
            # the origin for the image coordinate system is at the top-left corner
            v_image[1, :] = self.img_H - v_image[1, :]
        v_shape[-1] = 2
        v_image = v_image.transpose().reshape(v_shape)
        return v_image

    def world_space_point3D(self, point2Ds: tuple or list or np.ndarray, depths: float or np.ndarray, is_2D_origin_bottom_left):
        """
        Return the 3D points in world space
        :param point2Ds: 2D points in the image space:
            list or tuple: [x, y]
            np.ndarray: shape(..., 2)
        :param depths: depths of the points
            list or tuple: [depth]
            np.ndarray: shape(..., 1)
        :param is_2D_origin_bottom_left: if the origin for the given point2D is at the bottom-left corner
        :return: 3D points in world space
        """
        if isinstance(point2Ds, list) or isinstance(point2Ds, tuple):
            point2Ds = np.array(point2Ds, dtype=np.float32).flatten()
            depths = np.array(depths, dtype=np.float32).flatten()
            assert len(point2Ds) == 2, f"Error length: {len(point2Ds)}"
            assert len(depths) == 1, f"Error length: {len(depths)}"
            point2Ds = point2Ds.reshape(1, 2)
            depths = depths.reshape(1, 1)
        assert point2Ds.shape[-1] == 2, f"Error shape: {point2Ds.shape}"
        assert depths.shape[-1] == 1, f"Error shape: {depths.shape}"
        shape = list(point2Ds.shape)
        assert point2Ds.shape[:-1] == depths.shape[:-1], f"Mismatch shape: {point2Ds.shape}, {depths.shape}"
        assert point2Ds.shape[-1] == 2 and depths.shape[-1] == 1, f"Error shape: {point2Ds.shape}, {depths.shape}"
        point2Ds = point2Ds.reshape(-1, 2)
        depths = depths.reshape(-1, 1)
        if not is_2D_origin_bottom_left:  # the origin for the given point2D is at the top-left corner
            point2Ds[:, 1] = self.img_H - point2Ds[:, 1]
        pts = np.concatenate([point2Ds, np.ones((point2Ds.shape[0], 1))], axis=1)
        pts = pts.transpose()  # 3 X N
        depths = depths.transpose()  # 1 X N
        pts = np.linalg.inv(self.intrinsic_3x3()) @ pts * depths
        pts = pts.transpose()  # N X 3
        pts = self.p.orient.apply(pts) + self.p.loc.reshape(1, 3)
        shape[-1] += 1
        return pts.reshape(shape)

    def transform(self, R, T):
        """Transform the camera by R and T
        :param R: Rotation
        :param T: Translation
        :return: Camera after transformation"""
        return Camera(self.p.transform(R, T), (self.img_W, self.img_H), self.FOV_deg)

    def get_R_T(self):
        """Return R and T of the camera extrinsic parameters"""
        return self.R, self.T

    def get_look_at(self):
        """Return the look-at direction of the camera in world space"""
        return self.p.orient.apply(self.DEFAULT_LOOKAT)

    def intrinsic_3x3(self):
        """Return the 3x3 intrinsic matrix of the camera"""
        return self._intrinsic_matrix

    def intrinsic_3x4(self):
        """Return the 3x4 intrinsic matrix of the camera"""
        return np.concatenate([self._intrinsic_matrix, np.zeros((3, 1))], axis=1)

    def extrinsic_3x4(self):
        """Return the 3x4 extrinsic matrix of the camera"""
        return self._extrinsic_matrix

    def extrinsic_4x4(self):
        """Return the 4x4 extrinsic matrix of the camera"""
        return np.concatenate([self._extrinsic_matrix, np.array([[0, 0, 0, 1]])], axis=0)
