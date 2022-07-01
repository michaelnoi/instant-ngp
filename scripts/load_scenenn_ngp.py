import imp
import os
import copy
import shutil
import random
import json
from tkinter import image_names
import h5py
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


def load_scenenn_trajectory(camera_path):
    pos_file = os.listdir(camera_path)
    pos_file = [f for f in pos_file if f.endswith('npy')]
    if len(pos_file) == 0:
        pos_file = [f for f in pos_file if f.endswith('txt')]

    transform_matrices = []
    for f in pos_file:
        if f.endswith('npy'):
            xform = np.load(os.path.join(camera_path, f))
        else:
            xform = np.loadtxt(os.path.join(camera_path, f))

        xform[0:3,2] *= -1 # flip the y and z axis
        xform[0:3,1] *= -1
        
        transform_matrices.append(xform)

    return transform_matrices, pos_file


def get_image_count(img_path):
    img_names = os.listdir(img_path)
    img_names = [x for x in img_names if x.endswith('tonemap.jpg')]
    return len(img_names)


def collect_scenenn_images(img_path, img_prefix='', output_path='images'):
    img_names = os.listdir(img_path)
    img_names = [x for x in img_names if x.endswith('png')]
    new_img_names = [img_prefix + x for x in img_names]

    os.makedirs(output_path, exist_ok=True)
    for i in range(len(img_names)):
        shutil.copy(os.path.join(img_path, img_names[i]), os.path.join(output_path, new_img_names[i]))

    return new_img_names


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


def transform_scenenn_trajectory(transform_matrices):
    avglen = 0.
    for f in transform_matrices:
        avglen += np.linalg.norm(f[0:3,3])
    avglen /= len(transform_matrices)
    # print("avg camera distance from origin", avglen)

    for f in transform_matrices:
        f[0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    # offsets = np.array([x[:3, 3] for x in transform_matrices])
    # mean_offset = offsets.mean(axis=0)
    # dists = np.linalg.norm(offsets - mean_offset, axis=1)
    # mean_dist = dists.mean()

    # for i in range(len(transform_matrices)):
    #     transform_matrices[i][:3, 3] -= mean_offset
    #     transform_matrices[i][:3, 3] /= (mean_dist / 2)

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in transform_matrices:
        mf = f[0:3,:]
        for g in transform_matrices:
            mg = g[0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.01:
                totp += p*w
                totw += w

    totp /= totw
    # print("looking at", totp) # the cameras are looking at totp
    for f in transform_matrices:
        f[0:3,3] -= totp

    xform = np.eye(4)
    xform[[0, 1, 2], [0, 1, 2]] = 4.0 / avglen
    xform[:3, 3] = -totp
    # print("transformation performed\n", xform)

    return xform


def get_bounding_boxes(obj_poses_path, transform, type='obb'):
    '''
    Type can be "obb" or "aabb"
    '''
    obbox = []
    aabb = []
    poses = []

    root = ET.parse(obj_poses_path).getroot()
    for child in root:
        tokens = [float(x) for x in child.attrib['obbox'].split()]
        aabb_tokens = [float(x) for x in child.attrib['aabbox'].split()]
        pose = [float(x) for x in child.attrib['local_pose'].split()]

        obbox.append(tokens)
        aabb.append(aabb_tokens)
        poses.append(pose)

    obbox = np.array(obbox)
    aabb = np.array(aabb)
    poses = np.array(poses)

    if type == 'obb':
        extents = obbox[:, 3:6]
        pos = obbox[:, :3]
        orientation = [R.from_quat(obbox[i, 6:]).as_matrix() for i in range(obbox.shape[0])]
        orientation = np.array(orientation)
    else:
        pos = aabb[:, :3]
        extents = aabb[:, 3:]
        orientation = [R.from_quat(poses[i]).as_matrix() for i in range(poses.shape[0])]
        orientation = np.array(orientation)

    assert extents.shape[0] == orientation.shape[0] == pos.shape[0]

    extents *= transform[[0, 1, 2], [0, 1, 2]]
    pos = np.matmul(pos, transform[:3, :3].T) + transform[:3, 3]

    return extents, orientation, pos


def create_validation_json(json_head, img_names, xforms, num_train_samples, num_val_samples):
    json_dict = copy.deepcopy(json_head)
    num_train_samples = min(num_train_samples, len(img_names))
    train_samples = random.sample(range(len(img_names)), num_train_samples)

    for i in range(len(train_samples)):
        prefix = img_names[i].split('.')[0]
        json_dict["frames"].append({
            "file_path": img_names[i],
            "transform_matrix": xforms[prefix].tolist()
        })

    ext = img_names[0].split('.')[-1]
    for i in range(num_val_samples):
        views = np.random.choice(range(len(img_names)), 2, replace=False)
        views = [image_names[i].split('.')[0] for i in views]
        interpolated = np.eye(4)
        interpolated[:3, :3] = xforms[views[0]][:3, :3]
        interpolated[:3, 3] = (xforms[views[0]][:3, 3] + xforms[views[1]][:3, 3]) * 0.5

        json_dict["frames"].append({
            "file_path": f'val_{i}.{ext}',
            "transform_matrix": interpolated.tolist()
        })

    return json_dict


def create_json_from_scenenn(scene_path, output_path='./', obj_poses_path=None, bbox_type='obb', 
                             val_output_path=None, num_train_samples=10, num_val_samples=10):
    os.makedirs(output_path, exist_ok=True)

    transform_matrices, pos_names = load_scenenn_trajectory(scene_path)
    img_names = collect_scenenn_images(scene_path, '', os.path.join(output_path, 'images'))

    height, width = 480, 640
    focal_length = 544.47329
    cx, cy = 320, 240

    fov_x = 2 * np.arctan(width / (2 * focal_length))
    fov_y = 2 * np.arctan(height / (2 * focal_length))

    xform = transform_scenenn_trajectory(transform_matrices)
    if obj_poses_path is not None:
        extents, orientation, pos = get_bounding_boxes(obj_poses_path, xform, bbox_type)

    json_dict = {
        "camera_angle_x": fov_x,
        "camera_angle_y": fov_y,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "aabb_scale": 4,
        "frames": [],
        "bounding_boxes": [],
    }

    xform_dict = {pos_names[i].split('.')[0]: transform_matrices[i] for i in range(len(pos_names))}

    if val_output_path is not None:
        val_json = create_validation_json(json_dict, img_names, xform_dict, num_train_samples, num_val_samples)

    for i in range(len(img_names)):
        prefix = img_names[i].split('.')[0]
        json_dict["frames"].append({
            "file_path": os.path.join('images', img_names[i]),
            "transform_matrix": xform_dict[prefix].tolist()
        })

    if obj_poses_path is not None:
        for i in range(len(extents)):
            json_dict["bounding_boxes"].append({
                "extents": extents[i].tolist(),
                "orientation": orientation[i].tolist(),
                "position": pos[i].tolist(),
            })

    if val_output_path is not None:
        val_json['bounding_boxes'] = json_dict['bounding_boxes']
        os.makedirs(val_output_path, exist_ok=True)
        with open(os.path.join(val_output_path, 'val_transforms.json'), 'w') as f:
            json.dump(val_json, f, indent=2)

    with open(os.path.join(output_path, 'transforms.json'), 'w') as f:
        json.dump(json_dict, f, indent=2)

    return json_dict
    