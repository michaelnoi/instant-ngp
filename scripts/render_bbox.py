import cv2
import numpy as np


def nerf_matrix_to_ngp(nerf_matrix, scale, offset):
    """
    Convert a matrix from the NeRF coordinate system to the ngp coordinate system.
    Check the code in nerf_loader.h for more details.
    """
    ngp_matrix = np.copy(nerf_matrix)
    ngp_matrix[:, 1:3] *= -1
    ngp_matrix[:, -1] = ngp_matrix[:, -1] * scale + offset

    tmp = np.copy(ngp_matrix[0, :])
    ngp_matrix[0, :] = ngp_matrix[1, :]
    ngp_matrix[1, :] = ngp_matrix[2, :]
    ngp_matrix[2, :] = tmp

    return ngp_matrix


def get_bbox_corners(bbox, scale, offset):
    extents = np.array(bbox['extents']) * scale
    orientation = np.array(bbox['orientation'])
    position = np.array(bbox['position'])
    
    xform = np.zeros((3, 4))
    xform[:3, :3] = orientation
    xform[:3, 3] = position
    ngp_xform = nerf_matrix_to_ngp(xform, scale, offset)

    corners = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1]
    ]) * 0.5 * np.expand_dims(extents, 0)

    corners = ngp_xform[:3, :3] @ corners.T + np.expand_dims(ngp_xform[:3, 3], -1)
    return corners.T


def add_bbox_line(img, world2proj, a, b, color=(0, 0, 255), thickness=2):
    ha = np.array([a[0], a[1], a[2], 1]).reshape(4, 1)
    hb = np.array([b[0], b[1], b[2], 1]).reshape(4, 1)
    ha = np.squeeze(world2proj @ ha).T
    hb = np.squeeze(world2proj @ hb).T

    if ha[3] <= 0 or hb[3] <= 0:
        return

    aa = tuple((ha[:2] / ha[3]).astype(np.int))
    bb = tuple((hb[:2] / hb[3]).astype(np.int))

    h, w, c = img.shape
    cv2.line(img, aa, bb, color, thickness)


def render_bbox_overlay(img, world2proj, scale, offset, bbox, color=(0, 0, 255), thickness=2):
    corners = get_bbox_corners(bbox, scale, offset)

    for i in range(4):
        add_bbox_line(img, world2proj, corners[i, :], corners[(i + 1) % 4, :], color, thickness)
        add_bbox_line(img, world2proj, corners[i + 4, :], corners[(i + 1) % 4 + 4, :], color, thickness)
        add_bbox_line(img, world2proj, corners[i, :], corners[i + 4, :], color, thickness)

