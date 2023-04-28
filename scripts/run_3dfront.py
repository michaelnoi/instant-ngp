import subprocess
import argparse
import os
import random
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm


def create_validation_json(json_train, num_train_samples, num_val_samples):
	'''
	Create json for validation. Use train views and interpolate new views.
	'''
	json_dict = deepcopy(json_train)
	frames = json_train['frames']
	json_dict['frames'] = []

	num_train_samples = min(num_train_samples, len(frames))
	train_samples = random.sample(range(len(frames)), num_train_samples)

	for i in train_samples:
		json_dict["frames"].append(frames[i])

	ext = frames[0]['file_path'].split('.')[-1]
	for i in range(num_val_samples):
		views = np.random.choice(range(len(frames)), 2, replace=False)
		interpolated = np.eye(4)

		xforms1 = np.array(frames[views[0]]['transform_matrix'])
		xforms2 = np.array(frames[views[1]]['transform_matrix'])

		interpolated[:3, :3] = xforms1[:3, :3]
		interpolated[:3, 3] = (xforms2[:3, 3] + xforms1[:3, 3]) * 0.5

		json_dict["frames"].append({
			"file_path": f'val_{i}.{ext}',
			"transform_matrix": interpolated.tolist()
		})

	return json_dict


def parse_args():
	parser = argparse.ArgumentParser(description='Train ngp nerf with 3D FRONT data, output rendering results.')

	parser.add_argument('--data_path', default='', required=True, help='The path to the 3D FRONT folder containing all scenes')
	parser.add_argument('--run_script_path', default='./run.py', type=str, help='The path to the run.py script')
	parser.add_argument('--output_path', default='./results', help='The path to the output folder')
	parser.add_argument('--save_snapshot', default='model.msgpack', 
						help='Save this snapshot after training. recommended extension: .msgpack')

	parser.add_argument('--samples_from_train', default=10, type=int, help='The number of validation views from the training set')
	parser.add_argument('--num_val_samples', default=10, type=int, help='The number of interpolated validation views')
	parser.add_argument('--screenshot_spp', type=int, default=1, help='Number of samples per pixel in screenshots.')
	parser.add_argument('--n_steps', type=int, default=30000, help='Number of steps to train for before quitting.')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	os.makedirs(args.output_path, exist_ok=True)

	scenes = os.listdir(args.data_path)
	scenes = [s for s in scenes if os.path.isdir(os.path.join(args.data_path, s))]
	scenes = sorted(scenes)

	failed_scenes = []

	print('Training and rendering...')
	for idx, scene in enumerate(scenes):

		if scene in os.listdir(args.output_path):
			print('Scene {} already trained, skipping...'.format(scene))
			continue

		scene_path = os.path.join(args.data_path, scene, 'train')
		files = os.listdir(scene_path)

		if 'transforms.json' not in files:
			print('No transforms.json found for scene: {}'.format(scene))
			continue

		print('Training scene {}/{}: {}'.format(idx+1, len(scenes), scene))

		os.makedirs(os.path.join(args.output_path, scene, 'val'), exist_ok=True)

		with open(os.path.join(scene_path, 'transforms.json'), 'r') as f:
			json_train = json.load(f)
			if 'frames' not in json_train or len(json_train['frames']) < 100:
				print(f'Not enough poses for scene {scene}, skipping...')
				continue
				
			json_val = create_validation_json(json_train, args.samples_from_train, args.num_val_samples)

		with open(os.path.join(args.output_path, scene, 'val', 'val_transforms.json'), 'w') as f:
			json.dump(json_val, f, indent=2)

		save_snapshot = ''
		if args.save_snapshot:
			save_snapshot = os.path.join(args.output_path, scene, args.save_snapshot)

		arg_list = [
			'python', str(args.run_script_path), 
			'--scene', scene_path,
			'--mode', 'nerf',
			'--save_snapshot', str(save_snapshot),
			'--screenshot_transforms', str(os.path.join(args.output_path, scene, 'val', 'val_transforms.json')),
			'--screenshot_dir', str(os.path.join(args.output_path, scene, 'val', 'screenshots')),
			'--screenshot_spp', str(args.screenshot_spp),
			'--n_steps', str(args.n_steps),
			'--width', str(int(json_train['w'])),
			'--height', str(int(json_train['h'])),
			'--save_log', str(os.path.join(args.output_path, scene, 'log'))
		]

		try:
			subprocess.run(arg_list)
		except:
			print('Failed to train scene: {}'.format(scene))
			failed_scenes.append(scene)

	print('Failed scenes: {}'.format(failed_scenes))
