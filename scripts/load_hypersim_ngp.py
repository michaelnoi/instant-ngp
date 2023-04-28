import os
import copy
import shutil
import random
import json
import h5py
import numpy as np
import pandas as pd


def load_hypersim_trajectory(camera_path):
    orientation_filename = os.path.join(camera_path, 'camera_keyframe_orientations.hdf5')
    pos_filename = os.path.join(camera_path, 'camera_keyframe_positions.hdf5')

    orientation_file = h5py.File(orientation_filename, 'r')
    pos_file = h5py.File(pos_filename, 'r')

    orientation_data = orientation_file['dataset']
    pos_data = pos_file['dataset']

    transform_matrices = []
    for i in range(orientation_data.shape[0]):
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = orientation_data[i]
        affine_matrix[:3, 3] = pos_data[i]
        transform_matrices.append(affine_matrix)

    return transform_matrices


def get_image_count(img_path):
    img_names = os.listdir(img_path)
    img_names = [x for x in img_names if x.endswith('tonemap.jpg')]
    return len(img_names)


def collect_hypersim_images(img_path, img_prefix='', output_path='images'):
    img_names = os.listdir(img_path)
    img_names = [x for x in img_names if x.endswith('tonemap.jpg')]
    new_img_names = [img_prefix + x for x in img_names]

    os.makedirs(output_path, exist_ok=True)
    for i in range(len(img_names)):
        shutil.copy(os.path.join(img_path, img_names[i]), os.path.join(output_path, new_img_names[i]))

    return new_img_names


def get_hypersim_intrinsics(metadata_path, scene_name):
    metadata_df = pd.read_csv(metadata_path)
    params = metadata_df[metadata_df['scene_name'] == scene_name]
    fov_x = params['camera_physical_fov'].values[0]
    if np.isnan(fov_x):
        fov_x = 1.0471975803375244      # default

    height = params['settings_output_img_height'].values[0]
    width = params['settings_output_img_width'].values[0]
    focal_length = width / (2 * np.tan(fov_x / 2))
    fov_y = 2 * np.arctan(height / (2 * focal_length))

    return height, width, fov_y, fov_x, focal_length


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


def transform_hypersim_trajectory(transform_matrices):
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


def get_bounding_boxes(mesh_path, transform):
    extents_file = h5py.File(os.path.join(mesh_path, 'metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5'), 'r')
    orientation_file = h5py.File(os.path.join(mesh_path, 'metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5'), 'r')
    pos_file = h5py.File(os.path.join(mesh_path, 'metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5'), 'r')

    extents_data = extents_file['dataset']
    orientation_data = orientation_file['dataset']
    pos_data = pos_file['dataset']

    extents = np.array(extents_data)
    orientation = np.array(orientation_data)
    pos = np.array(pos_data)

    ext_inf_filter = np.isinf(extents)
    ori_inf_filter = np.isinf(orientation)
    
    extents = extents[~ext_inf_filter].reshape(-1, 3)
    orientation = orientation[~ori_inf_filter].reshape(-1, 3, 3)
    pos = pos[~ext_inf_filter].reshape(-1, 3)

    assert extents.shape[0] == orientation.shape[0] == pos.shape[0]

    extents *= transform[[0, 1, 2], [0, 1, 2]]
    pos = np.matmul(pos, transform[:3, :3].T) + transform[:3, 3]

    return extents, orientation, pos


def create_validation_json(json_head, img_names, xforms, num_train_samples, num_val_samples):
    json_dict = copy.deepcopy(json_head)
    num_train_samples = min(num_train_samples, len(img_names))
    train_samples = random.sample(range(len(img_names)), num_train_samples)

    for i in train_samples:
        json_dict["frames"].append({
            "file_path": img_names[i],
            "transform_matrix": xforms[i].tolist()
        })

    ext = img_names[0].split('.')[-1]
    for i in range(num_val_samples):
        views = np.random.choice(range(len(img_names)), 2, replace=False)
        interpolated = np.eye(4)
        interpolated[:3, :3] = xforms[views[0]][:3, :3]
        interpolated[:3, 3] = (xforms[views[0]][:3, 3] + xforms[views[1]][:3, 3]) * 0.5

        json_dict["frames"].append({
            "file_path": f'val_{i}.{ext}',
            "transform_matrix": interpolated.tolist()
        })

    return json_dict


def create_json_from_hypersim(scene_path, metadata_path, use_all_cameras=False, num_samples=None, 
                              output_path='./', val_output_path=None, num_train_samples=20, 
                              num_val_samples=20):
    scene_name = os.path.basename(scene_path)
    os.makedirs(output_path, exist_ok=True)

    camera_metadata = pd.read_csv(os.path.join(scene_path, '_detail', 'metadata_cameras.csv'))
    camera_names = camera_metadata['camera_name'].tolist()
    camera_names = [c for c in camera_names 
                    if os.path.exists(os.path.join(scene_path, '_detail', c)) 
                    and os.path.exists(os.path.join(scene_path, 'images', 'scene_{}_final_preview'.format(c)))]

    camera_paths = []
    img_paths = []
    for camera in camera_names:
        camera_path = os.path.join(scene_path, '_detail', camera)
        img_path = os.path.join(scene_path, 'images', 'scene_{}_final_preview'.format(camera))
        if os.path.exists(camera_path) and os.path.exists(img_path):
            camera_paths.append(camera_path)
            img_paths.append(img_path)

    mesh_path = os.path.join(scene_path, '_detail', 'mesh')

    if not use_all_cameras:
        img_cnts = [get_image_count(x) for x in img_paths]
        cam_idx = np.argmax(img_cnts)       # select the camera with the most images
        camera_path = camera_paths[cam_idx]
        img_path = img_paths[cam_idx]
        transform_matrices = load_hypersim_trajectory(camera_path)
        img_names = collect_hypersim_images(img_path, output_path=os.path.join(output_path, 'images'))
    else:
        transform_matrices = []
        num_xforms = []
        img_names = []

        for i in range(len(camera_names)):
            imgs = collect_hypersim_images(img_paths[i], camera_names[i] + '.', 
                                           output_path=os.path.join(output_path, 'images'))

            xforms = load_hypersim_trajectory(camera_paths[i])
            # if len(imgs) != len(xforms):
            #     # Some images are missing, filter out corresponding transforms
            #     labels = [int(img.split('.')[2]) for img in imgs]
            #     xforms = [xform for i, xform in enumerate(xforms) if i in labels]

            num_xforms.append(len(transform_matrices))
            transform_matrices += xforms
            img_names += imgs

    height, width, fov_y, fov_x, focal_length = get_hypersim_intrinsics(metadata_path, scene_name)

    if num_samples is not None:
        sampled = np.random.randint(0, len(transform_matrices), num_samples)
        transform_matrices = [transform_matrices[i] for i in sampled]
        img_names = [img_names[i] for i in sampled]

    # TODO: Only compute xform based on transform matrices which actually have corresponding images
    xform = transform_hypersim_trajectory(transform_matrices)
    extents, orientation, pos = get_bounding_boxes(mesh_path, xform)

    json_dict = {
        "camera_angle_x": fov_x,
        "camera_angle_y": fov_y,
        "fl_x": focal_length,
        "fl_y": focal_length,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": width / 2,
        "cy": height / 2,
        "w": width,
        "h": height,
        "aabb_scale": 4,
        "frames": [],
        "bounding_boxes": [],
    }

    if val_output_path is not None:
        val_json = create_validation_json(json_dict, img_names, transform_matrices, num_train_samples, num_val_samples)

    for i in range(len(img_names)):
        frame_idx = int(img_names[i].split('.')[1]) if not use_all_cameras else int(img_names[i].split('.')[2])
        if use_all_cameras:
            camera_idx = camera_names.index(img_names[i].split('.')[0])
            frame_idx += num_xforms[camera_idx]

        json_dict["frames"].append({
            "file_path": os.path.join('images', img_names[i]),
            "transform_matrix": transform_matrices[frame_idx].tolist()
        })

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
    