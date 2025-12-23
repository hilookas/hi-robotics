from fractions import Fraction
import math


def fov_from_f(fy, height):
    fov_y = math.atan(height / 2 / fy) * 2 / math.pi * 180
    return fov_y


def f_from_fov(fov_y_deg, height):
    fy = height / 2 / math.tan(fov_y_deg * math.pi / 180 / 2)
    return fy


def intrinsic_from_fx_fy_cx_cy(fx, fy, cx, cy):
    return [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ]


def intrinsic_from_f(f, width, height):
    fx = fy = f
    cx = width / 2
    cy = height / 2
    return intrinsic_from_fx_fy_cx_cy(fx, fy, cx, cy)


def crop_scale(source_height: int, source_width: int, target_height: int, target_width: int) -> tuple[Fraction, Fraction, Fraction]:
    if Fraction(source_width, source_height) > Fraction(target_width, target_height):
        # crop width
        scale_ratio = Fraction(target_height, source_height)

        target_height / source_height
        crop_source_height_start = Fraction(0, 1)
        crop_source_width_start = (source_width - target_width / scale_ratio) / 2
    else:
        # crop height
        scale_ratio = Fraction(target_width, source_width)
        crop_source_width_start = Fraction(0, 1)
        crop_source_height_start = (source_height - target_height / scale_ratio) / 2

    return scale_ratio, crop_source_height_start, crop_source_width_start


def scaled_intrinsic(intrinsic: list[list[float]], scale_ratio: Fraction | float, crop_source_height_start: Fraction | float = 0.0, crop_source_width_start: Fraction | float = 0.0) -> list[list[float]]:
    import numpy as np
    scaled_intrinsic = np.array(intrinsic).copy()
    scaled_intrinsic[0, 2] = scaled_intrinsic[0, 2] - crop_source_width_start # cx
    scaled_intrinsic[1, 2] = scaled_intrinsic[1, 2] - crop_source_height_start # cy
    scaled_intrinsic[0, :] = scaled_intrinsic[0, :] * scale_ratio # fx cx
    scaled_intrinsic[1, :] = scaled_intrinsic[1, :] * scale_ratio # fy cy
    return scaled_intrinsic.tolist()


def xyz_from_uvd(uvd, intrinsic_matrix, depth_scale):
    u, v, d = uvd
    # https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)


def uvd_from_xyz(xyz, intrinsic_matrix, depth_scale):
    x, y, z = xyz
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    d = z * depth_scale
    return (u, v, d)