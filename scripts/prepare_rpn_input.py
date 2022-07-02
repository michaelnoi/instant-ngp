import sys

pyngp_path = '../build'
sys.path.append(pyngp_path)
import pyngp as ngp

import numpy as np
import json
import argparse
import os

from tqdm import tqdm


def construct_grid(res_x, res_y, res_z):
    x = np.linspace(-res_x, res_x, res_x)
    y = np.linspace(-res_y, res_y, res_y)
    z = np.linspace(-res_z, res_z, res_z)

    x /= max(res_x, res_y, res_z)
    y /= max(res_x, res_y, res_z)
    z /= max(res_x, res_y, res_z)

    grid = []
    for i in range(res_z):
        for j in range(res_y):
            for k in range(res_x):
                grid.append([x[k], y[j], z[i]])

    return np.array(grid)


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def get_ngp_obj_bounding_box(xform, extent):
    corners = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1],
    ], dtype=float).T

    corners *= np.expand_dims(extent, 1) * 0.5
    corners = xform[:, :3] @ corners + xform[:, 3, None]

    return np.min(corners, axis=1), np.max(corners, axis=1)


def get_scene_bounding_box(dataset, json_dict, margin=0.1):
    min_pt = []
    max_pt = []

    for obj in json_dict['bounding_boxes']:
        extent = np.array(obj['extents'])
        orientation = np.array(obj['orientation'])
        position = np.array(obj['position'])

        xform = np.hstack([orientation, np.expand_dims(position, 1)])
        xform = dataset.nerf_matrix_to_ngp(xform)
        extent *= dataset.scale

        min_pt_, max_pt_ = get_ngp_obj_bounding_box(xform, extent)
        min_pt.append(min_pt_)
        max_pt.append(max_pt_)

    min_pt = np.array(min_pt)
    max_pt = np.array(max_pt)

    min_pt = np.min(min_pt, axis=0)
    max_pt = np.max(max_pt, axis=0)

    enlarging_amt = (max_pt - min_pt) * margin
    min_pt -= enlarging_amt
    max_pt += enlarging_amt

    return min_pt, max_pt


def get_scene_config(path, testbed, use_dynamic_res=False, margin=0.1):
    dataset = testbed.nerf.training.dataset

    with open(path) as f:
        json_dict = json.load(f)
        cam_matrices = [np.array(x['transform_matrix']) for x in json_dict['frames']]
        min_pt, max_pt = get_scene_bounding_box(dataset, json_dict, margin)
        scene_bbox = ngp.BoundingBox(min_pt, max_pt).intersection(testbed.aabb)

        if not use_dynamic_res:
            min_pt, max_pt = min(scene_bbox.min), max(scene_bbox.max)
            scene_bbox = ngp.BoundingBox([min_pt] * 3, [max_pt] * 3)

    ngp_cams = [dataset.nerf_matrix_to_ngp(cam_matrix[:-1,:]) for cam_matrix in cam_matrices]
    view_dirs = [cam[:, :3] @ np.array([0, 0, 1]) for cam in ngp_cams]

    return scene_bbox, view_dirs, ngp_cams


def get_rgbsigma(max_res, testbed, view_dir, scene_bbox=None, use_dynamic_res=False):
    res = [max_res] * 3

    if use_dynamic_res:
        assert scene_bbox is not None
        diag = scene_bbox.max - scene_bbox.min
        res = diag / diag.max() * max_res
        res = np.array(res, dtype=int)
        res = np.minimum(res, max_res)

    if scene_bbox is not None:
        density = testbed.compute_density_on_grid(res, scene_bbox)
        rgba = testbed.compute_rgba_on_grid(res, view_dir, scene_bbox)
    else:
        density = testbed.compute_density_on_grid(res)
        rgba = testbed.compute_rgba_on_grid(res, view_dir)

    rgba[:, 3] = density.squeeze()
    return res, rgba


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RGB and density from a trained NeRF")

    parser.add_argument("--hypersim_path", default="", help="The path to the scene.")
    parser.add_argument("--snapshot_name", default="", help="Name of the snapshot.")
    parser.add_argument("--output_dir", default="", help="The path to the output directory.")
    parser.add_argument("--max_res", default=256, type=int, help="The maximum resolution of the output.")
    parser.add_argument("--use_dynamic_res", default=False, action="store_true", help="Use different resolutions for each dimension.")
    parser.add_argument("--margin", default=0.1, type=float, help="The margin added to the scene bounding box.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    scene_exclusion_list = []

    os.makedirs(args.output_dir, exist_ok=True)

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    dataset = testbed.nerf.training.dataset

    scenes = os.listdir(args.hypersim_path)
    scenes = [s for s in scenes if os.path.isdir(os.path.join(args.hypersim_path, s))]

    for scene in tqdm(scenes):
        if scene in scene_exclusion_list:
            continue

        testbed.load_training_data(os.path.join(args.hypersim_path, scene, 'train'))
        testbed.load_snapshot(os.path.join(args.hypersim_path, scene, 'train', args.snapshot_name))

        json_path = os.path.join(args.hypersim_path, scene, 'train', 'transforms.json')
        scene_bbox, view_dirs, ngp_cams = get_scene_config(json_path, testbed, args.use_dynamic_res, args.margin)

        rgbsigma_mean = None
        for i, view_dir in enumerate(view_dirs):
            res, rgbsigma = get_rgbsigma(args.max_res, testbed, view_dir, scene_bbox, args.use_dynamic_res)
            if rgbsigma_mean is None:
                rgbsigma_mean = rgbsigma
            else:
                rgbsigma_mean = (rgbsigma_mean * i + rgbsigma) / (i + 1)

        output_path = os.path.join(args.output_dir, scene + '.npz')

        np.savez_compressed(output_path, rgbsigma = rgbsigma, resolution=res,
                            bbox_min=scene_bbox.min, bbox_max=scene_bbox.max,
                            scale=dataset.scale, offset=dataset.offset,
                            from_mitsuba=dataset.from_mitsuba)

