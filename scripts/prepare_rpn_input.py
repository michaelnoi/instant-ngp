import sys
import os

# if os.path.exists('build'):
#     sys.path.append('./build')
# else:
#     sys.path.append('../build')

sys.path.append('../instant-ngp/build')

import pyngp as ngp

import numpy as np
import json
import argparse

from tqdm import tqdm
import torch
from sh_proj import ProjectFunctionNeRF


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


def transform_to_ngp_bbox(bbox_raw, dataset):
    """ 
        Transform a bounding box from the raw dataset to the ngp coordinate. (for 3dfront)
        Input:
            bbox_raw: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            dataset: ngp.Dataset
        Return:
            min_pt: [min_x, min_y, min_z]
            max_pt: [max_x, max_y, max_z]
    """
    bbox_raw = np.array(bbox_raw)
    min_pt_raw, max_pt_raw = bbox_raw[0], bbox_raw[1]
    extent = max_pt_raw - min_pt_raw
    position = (min_pt_raw+max_pt_raw)/2

    xform = np.hstack([np.eye(3, 3), np.expand_dims(position, 1)])
    xform = dataset.nerf_matrix_to_ngp(xform)
    extent *= dataset.scale
    min_pt, max_pt = get_ngp_obj_bounding_box(xform, extent)

    return min_pt, max_pt


def get_scene_bounding_box(dataset, json_dict, margin=0.1, angle=None):

    if 'bounding_boxes' in json_dict and len(json_dict['bounding_boxes']) > 0:
        # Use object bounding boxes
        print('Estimating scene bounding box using object bounding boxes')
        min_pt = []
        max_pt = []

        for obj in json_dict['bounding_boxes']:
            extent = np.array(obj['extents'])
            orientation = np.array(obj['orientation'])
            position = np.array(obj['position'])

            xform = np.hstack([orientation, np.expand_dims(position, 1)])

            # xform = dataset.nerf_matrix_to_ngp(xform)
            # extent *= dataset.scale

            # if angle is not None:
            #     rot = np.array([
            #         [np.cos(angle), 0, np.sin(angle)],
            #         [0, 1, 0],
            #         [-np.sin(angle), 0, np.cos(angle)]
            #     ]).T
            #     xform = rot @ xform

            min_pt_, max_pt_ = get_ngp_obj_bounding_box(xform, extent)
            min_pt.append(min_pt_)
            max_pt.append(max_pt_)

        min_pt = np.array(min_pt)
        max_pt = np.array(max_pt)
        min_pt = np.min(min_pt, axis=0)
        max_pt = np.max(max_pt, axis=0)

    else:
        # Estimate bounding box from the cameras
        print('Estimating scene bounding box using cameras')
        camera_pos = []
        for frame in json_dict['frames']:
            xform = np.array(frame['transform_matrix'])
            # xform = dataset.nerf_matrix_to_ngp(xform[:-1, :])
            # if angle is not None:
            #     rot = np.array([
            #         [np.cos(angle), 0, np.sin(angle)],
            #         [0, 1, 0],
            #         [-np.sin(angle), 0, np.cos(angle)]
            #     ]).T
            #     xform = rot @ xform

            camera_pos.append(xform[:3, 3])

        camera_pos = np.array(camera_pos)
        min_pt = np.min(camera_pos, axis=0)
        max_pt = np.max(camera_pos, axis=0)

    enlarging_amt = (max_pt - min_pt) * margin
    min_pt -= enlarging_amt
    max_pt += enlarging_amt

    print('Scene bounding box: ', min_pt, max_pt)

    xform = np.hstack([np.eye(3, 3), np.expand_dims(min_pt, 1)])
    ngp_min_pt = dataset.nerf_matrix_to_ngp(xform)[:, 3]
    xform = np.hstack([np.eye(3, 3), np.expand_dims(max_pt, 1)])
    ngp_max_pt = dataset.nerf_matrix_to_ngp(xform)[:, 3]

    return min_pt, max_pt, ngp_min_pt, ngp_max_pt


def get_scene_config(path, testbed, use_dynamic_res=False, margin=0.1, angle=None, dataset_type='hypersim'):
    dataset = testbed.nerf.training.dataset

    with open(path) as f:
        json_dict = json.load(f)
        cam_matrices = [np.array(x['transform_matrix']) for x in json_dict['frames']]
        if dataset_type == 'hypersim':
            ori_min_pt, ori_max_pt, min_pt, max_pt = get_scene_bounding_box(dataset, json_dict, margin, angle)
        elif dataset_type == '3dfront':
            min_pt, max_pt = transform_to_ngp_bbox(json_dict['room_bbox'], dataset)
        elif dataset_type == '3rscan':
            ori_min_pt, ori_max_pt, min_pt, max_pt = get_scene_bounding_box(dataset, json_dict, margin, angle)
        else:
            raise ValueError('Unknown dataset type: {}'.format(dataset_type))

        scene_bbox = ngp.BoundingBox(min_pt, max_pt).intersection(testbed.aabb)

        if not use_dynamic_res:
            min_pt, max_pt = min(scene_bbox.min), max(scene_bbox.max)
            scene_bbox = ngp.BoundingBox([min_pt] * 3, [max_pt] * 3)

    if dataset_type == 'hypersim' or dataset_type == '3rscan':
        with open(path, 'w') as f:
            json_dict['room_bbox'] = [ori_min_pt.tolist(), ori_max_pt.tolist()]
            json.dump(json_dict, f, indent=2)

    ngp_cams = [dataset.nerf_matrix_to_ngp(cam_matrix[:-1,:]) for cam_matrix in cam_matrices]
    view_dirs = [cam[:, :3] @ np.array([0, 0, 1]) for cam in ngp_cams]

    return scene_bbox, view_dirs, ngp_cams


def get_rgbsigma(max_res, testbed, view_dir, scene_bbox=None, use_dynamic_res=False, transform_density_to_alpha=False, angle=None):
    res = [max_res] * 3

    if use_dynamic_res:
        assert scene_bbox is not None
        diag = scene_bbox.max - scene_bbox.min
        res = diag / diag.max() * max_res
        res = np.array(res, dtype=int)
        res = np.minimum(res, max_res)
        inds = res % 2 > 0
        res[inds] += 1

    # print('Resolution:', res)

    if angle is None:
        if scene_bbox is not None:
            density = testbed.compute_density_on_grid(res, scene_bbox)
            rgba = testbed.compute_rgba_on_grid(res, view_dir, scene_bbox)
        else:
            density = testbed.compute_density_on_grid(res)
            rgba = testbed.compute_rgba_on_grid(res, view_dir)
    else:
        assert scene_bbox is not None
        render_aabb_to_local = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]).T
        density = testbed.compute_density_on_grid(res, scene_bbox, render_aabb_to_local)
        rgba = testbed.compute_rgba_on_grid(res, view_dir, scene_bbox, render_aabb_to_local)

    if transform_density_to_alpha:
        density = density_to_alpha(density)

    rgba[:, 3] = density.squeeze()
    return res, rgba


def process_scene(args, testbed, dataset, scene_path, model_path, output_path):
    # testbed.load_training_data(scene_path)
    testbed.load_snapshot(model_path)

    json_path = os.path.join(scene_path, args.transforms_filename)
    scene_bbox, view_dirs, ngp_cams = get_scene_config(json_path, testbed, args.use_dynamic_res, 
                                                       args.margin, dataset_type=args.dataset_type)

    rgbsigma_mean = None
    for i, view_dir in tqdm(enumerate(view_dirs)):
        res, rgbsigma = get_rgbsigma(max_res=args.max_res, testbed=testbed, view_dir=view_dir, 
                                     scene_bbox=scene_bbox, use_dynamic_res=args.use_dynamic_res, 
                                     transform_density_to_alpha=args.density_to_alpha, angle=None)
        if rgbsigma_mean is None:
            rgbsigma_mean = rgbsigma
        else:
            rgbsigma_mean = (rgbsigma_mean * i + rgbsigma) / (i + 1)

    np.savez_compressed(output_path, rgbsigma = rgbsigma, resolution=res,
                        bbox_min=scene_bbox.min, bbox_max=scene_bbox.max,
                        scale=dataset.scale, offset=dataset.offset,
                        from_mitsuba=dataset.from_mitsuba)


def process_scene_with_frustum_culling(args, testbed, dataset, scene_path, model_path, output_path):
    '''
    Mask out view dirs that are not in the frustum
    '''
    # testbed.load_training_data(scene_path)
    testbed.load_snapshot(model_path)

    json_path = os.path.join(scene_path, args.transforms_filename)
    scene_bbox, view_dirs, ngp_cams = get_scene_config(json_path, testbed, args.use_dynamic_res, 
                                                       args.margin, dataset_type=args.dataset_type)

    with open(json_path) as f:
        json_dict = json.load(f)
        focal = json_dict['fl_x']   # assume fl_x == fl_y, cx = w/2, cy = h/2
        width = json_dict['w']
        height = json_dict['h']

    rgb = None
    sample_cnt = None
    unmasked_rgb = None

    for i, view_dir in enumerate(view_dirs):
        res, rgbsigma = get_rgbsigma(args.max_res, testbed, view_dir, scene_bbox, args.use_dynamic_res)
        sigma = rgbsigma[:, 3]

        if rgb is None:
            # Always use the first view dir as some samples may be outside all frustums
            rgb = np.zeros_like(rgbsigma[:, :3])
            sample_cnt = np.zeros_like(sigma)
            unmasked_rgb = np.zeros_like(rgbsigma[:, :3])

            x = np.linspace(scene_bbox.min[0], scene_bbox.max[0], res[0], endpoint=True)
            y = np.linspace(scene_bbox.min[1], scene_bbox.max[1], res[1], endpoint=True)
            z = np.linspace(scene_bbox.min[2], scene_bbox.max[2], res[2], endpoint=True)

            z, y, x = np.meshgrid(z, y, x, indexing='ij')
            sample_pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # frustum culling
        cam = ngp_cams[i].copy()
        cam[:, :3] = cam[:, :3].T       # c2w to w2c
        cam[:, 3] = -cam[:, :3] @ cam[:, 3]

        cam_space_pos = cam @ np.concatenate([sample_pos, np.ones((sample_pos.shape[0], 1))], axis=-1).T
        cam_space_pos = cam_space_pos.T
        ori_z = cam_space_pos[:, 2]
        cam_space_pos = cam_space_pos[:, :2] / ori_z[:, None]

        # frustum culling
        xmin = -width / 2 / focal
        xmax = width / 2 / focal
        ymin = -height / 2 / focal
        ymax = height / 2 / focal

        mask = (cam_space_pos[:, 0] >= xmin) & (cam_space_pos[:, 0] <= xmax) & \
                (cam_space_pos[:, 1] >= ymin) & (cam_space_pos[:, 1] <= ymax) & \
                (ori_z > 0)

        rgb[mask] = (rgb[mask] * sample_cnt[mask, None] + rgbsigma[mask, :3]) / (sample_cnt[mask, None] + 1)
        sample_cnt[mask] += 1
        unmasked_rgb += rgbsigma[:, :3]

    # Use all view dirs for samples that are not in any frustum
    rgb[sample_cnt == 0] = unmasked_rgb[sample_cnt == 0] / len(view_dirs)
    rgbsigma = np.concatenate([rgb, sigma[:, None]], axis=-1)

    print(f'mean sample count: {np.mean(sample_cnt)}')

    np.savez_compressed(output_path, rgbsigma = rgbsigma, resolution=res,
                        bbox_min=scene_bbox.min, bbox_max=scene_bbox.max,
                        scale=dataset.scale, offset=dataset.offset,
                        from_mitsuba=dataset.from_mitsuba)


def process_scene_with_fixed_view_dirs(args, testbed, dataset, scene_path, model_path, output_path):
    # testbed.load_training_data(scene_path)
    testbed.load_snapshot(model_path)

    json_path = os.path.join(scene_path, args.transforms_filename)
    scene_bbox, _, ngp_cams = get_scene_config(json_path, testbed, args.use_dynamic_res, 
                                               args.margin, dataset_type=args.dataset_type)

    view_dirs = []
    for i in range(-1, 2):
        pitch = i * np.pi / 4
        y = np.sin(pitch)
        for j in range(6):
            yaw = j * np.pi / 3
            x = np.cos(pitch) * np.cos(yaw)
            z = np.cos(pitch) * np.sin(yaw)
            view_dirs.append(np.array([x, y, z]))

    rgbsigma_mean = None
    for i, view_dir in enumerate(view_dirs):
        res, rgbsigma = get_rgbsigma(args.max_res, testbed, view_dir, scene_bbox, args.use_dynamic_res)
        if rgbsigma_mean is None:
            rgbsigma_mean = rgbsigma
        else:
            rgbsigma_mean = (rgbsigma_mean * i + rgbsigma) / (i + 1)

    np.savez_compressed(output_path, rgbsigma = rgbsigma, resolution=res,
                        bbox_min=scene_bbox.min, bbox_max=scene_bbox.max,
                        scale=dataset.scale, offset=dataset.offset,
                        from_mitsuba=dataset.from_mitsuba)               


def project_nerf_to_sh(testbed, max_res, scene_bbox, sh_deg, sample_count, use_dynamic_res):
    """
    Returns:
        coeffs for rgb. [N, C * (sh_deg + 1)**2]
    """

    def _sperical_func(viewdirs):
        # viewdirs: [num_rays, 3]
        # raw_rgb: [num_points, num_rays, 3]
        # sigma: [num_points]
        raw_rgb = []
        for i in range(len(viewdirs)):
            res, rgbsigma = get_rgbsigma(max_res, testbed, viewdirs[i], scene_bbox, use_dynamic_res)
            rgbsigma = torch.from_numpy(rgbsigma).float()
            raw_rgb.append(rgbsigma[:, :3])
            sigma = rgbsigma[:, 3]

        raw_rgb = torch.stack(raw_rgb, dim=1)  # [num_points, num_rays, 3]
        
        return raw_rgb, sigma

    res = [max_res] * 3
    if use_dynamic_res:
        assert scene_bbox is not None
        diag = scene_bbox.max - scene_bbox.min
        res = diag / diag.max() * max_res
        res = np.array(res, dtype=int)
        res = np.minimum(res, max_res)

    num_points = np.prod(res)

    coeffs, sigma = ProjectFunctionNeRF(
        order=sh_deg,
        sperical_func=_sperical_func,
        batch_size=num_points,
        sample_count=sample_count,
        device='cpu')

    return coeffs.reshape([num_points, -1]), sigma, res


def process_scene_sh(args, testbed, dataset, scene_path, model_path, output_path):
    '''
    Extract features from the NeRF by projecting rgb to SH.
    '''
    # testbed.load_training_data(scene_path)
    testbed.load_snapshot(model_path)

    json_path = os.path.join(scene_path, args.transforms_filename)
    scene_bbox, view_dirs, ngp_cams = get_scene_config(json_path, testbed, args.use_dynamic_res, 
                                                       args.margin, dataset_type=args.dataset_type)

    coeffs, sigma, res = project_nerf_to_sh(testbed, args.max_res, scene_bbox, args.sh_deg, 
                                            args.sample_count, args.use_dynamic_res)

    coeffs = coeffs.numpy()
    sigma = sigma.numpy()

    rgbsigma = np.concatenate([coeffs, sigma[:, None]], axis=1)

    np.savez_compressed(output_path, rgbsigma=rgbsigma, resolution=res,
                        bbox_min=scene_bbox.min, bbox_max=scene_bbox.max,
                        scale=dataset.scale, offset=dataset.offset,
                        from_mitsuba=dataset.from_mitsuba)


def process_scene_with_rotation(args, testbed, dataset, scene_path, model_path, output_dir, num_bins):
    testbed.load_training_data(scene_path)
    testbed.load_snapshot(model_path)

    json_path = os.path.join(scene_path, args.transforms_filename)
    scene_name = os.path.basename(scene_path)

    angle_bins = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)

    for idx, angle in enumerate(tqdm(angle_bins)):
        scene_bbox, view_dirs, ngp_cams = get_scene_config(
            json_path, testbed, args.use_dynamic_res, args.margin, angle, args.dataset_type
        )

        rgbsigma_mean = None
        for i, view_dir in enumerate(view_dirs):
            res, rgbsigma = get_rgbsigma(
                args.max_res, testbed, view_dir, scene_bbox, args.use_dynamic_res, angle=angle
            )
            if rgbsigma_mean is None:
                rgbsigma_mean = rgbsigma
            else:
                rgbsigma_mean = (rgbsigma_mean * i + rgbsigma) / (i + 1)

        output_path = os.path.join(output_dir, f'{scene_name}_{idx}.npz')
        np.savez_compressed(output_path, rgbsigma = rgbsigma, resolution=res,
                            bbox_min=scene_bbox.min, bbox_max=scene_bbox.max,
                            scale=dataset.scale, offset=dataset.offset,
                            from_mitsuba=dataset.from_mitsuba,
                            angle=angle)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RGB and density from a trained NeRF")

    parser.add_argument("--dataset_type", choices=['hypersim', '3dfront', '3rscan'], required=True)
    parser.add_argument("--dataset_path", default="", help="The path to the scenes.")
    parser.add_argument("--model_dir", default="", help="The path to the models.")
    parser.add_argument("--snapshot_name", default="base.msgpack", help="Name of the snapshot.")
    parser.add_argument("--output_dir", default="", help="The path to the output directory.")
    parser.add_argument("--max_res", default=256, type=int, help="The maximum resolution of the output.")
    parser.add_argument("--use_dynamic_res", default=False, action="store_true", help="Use different resolutions for each dimension.")
    parser.add_argument("--margin", default=0.1, type=float, help="The margin added to the scene bounding box.")
    parser.add_argument("--density_to_alpha", default=False, action="store_true", help="Convert density to alpha.")
    parser.add_argument("--transforms_filename", default="transforms.json", help="The name of the transforms file.")

    parser.add_argument("--sample_count", default=1000, type=int, help="The number of samples used to project NeRF to SH.")
    parser.add_argument("--sh_deg", default=3, type=int, help="The degree of SH used to project NeRF to SH.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    dataset = testbed.nerf.training.dataset

    scenes = os.listdir(args.dataset_path)
    scenes = [s for s in scenes if os.path.isdir(os.path.join(args.dataset_path, s))]

    for scene in tqdm(scenes):
        scene_dir = os.path.join(args.dataset_path, scene, 'train')
        if 'transforms.json' not in os.listdir(scene_dir):
            print(f'transforms.json not found for {scene}')
            continue

        model_path = os.path.join(args.model_dir, scene, args.snapshot_name)
        if not os.path.exists(model_path):
            print(f'model {model_path} not exists')
            continue

        process_scene(
            args, testbed, dataset, scene_dir,
            model_path, os.path.join(args.output_dir, f'{scene}.npz'),
        )
