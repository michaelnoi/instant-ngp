import subprocess
import argparse
import os
import pandas as pd
from tqdm import tqdm

from load_hypersim_ngp import create_json_from_hypersim


def parse_args():
	parser = argparse.ArgumentParser(description="Create hypersim dataset and train with it, output rendering results.")

	parser.add_argument("--hypersim_path", default="", required=True, help="The path to the hypersim folder containing all scenes")

	parser.add_argument("--metadata_path", default="", required=True, help="The path to the metadata file containing all scenes")

	parser.add_argument("--run_script_path", default="", required=True, help="The path to the run.py script")

	parser.add_argument("--output_path", default="./results", help="The path to the output folder")

	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--samples_from_train", default=20, type=int, help="The number of validation views from the training set")

	parser.add_argument("--num_val_samples", default=20, type=int, help="The number of interpolated validation views")

	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")

	parser.add_argument("--use_all_cameras", action="store_true", help="Use all cameras in the scene, not just the first one")

	parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to use")

	parser.add_argument("--resume_from_stats", default="", help="Resume training from the previous stats file")

	parser.add_argument("--range", default="", help="Range of scenes to use")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()

	os.makedirs(args.output_path, exist_ok=True)

	stats_dict = {
		"scene_name": [],
		"num_train_views": [],
		"num_bboxes": [],
		"width": [],
		"height": [],
		"trained": [],
	}

	scene_sort_func = lambda x: int(x.split("_")[1]) * 100 + int(x.split("_")[2])

	scenes = os.listdir(args.hypersim_path)
	scenes = [s for s in scenes if os.path.isdir(os.path.join(args.hypersim_path, s))]
	scenes = sorted(scenes, key=scene_sort_func)

	if args.range:
		print('Using scene {}'.format(args.range))
		scene_range = range(int(args.range.split("-")[0]), int(args.range.split("-")[1]) + 1)
		scenes = [s for s in scenes if int(s.split('_')[1]) in scene_range]

	scenes_needs_training = [s for s in scenes]
	scenes_trained = []

	if args.resume_from_stats:
		stats_df = pd.read_csv(args.resume_from_stats)
		prev_stats = stats_df.to_dict('records')
		scenes_trained = [s["scene_name"] for s in prev_stats if s["trained"]]
		scenes_needs_training = [s for s in scenes if s not in scenes_trained]

	print("Converting hypersim scenes to json...")
	for scene in tqdm(scenes):
		scene_path = os.path.join(args.hypersim_path, scene)
		json_dict = create_json_from_hypersim(scene_path, args.metadata_path, 
											  output_path=os.path.join(args.output_path, scene, 'train'),
											  val_output_path=os.path.join(args.output_path, scene, 'val'),
											  use_all_cameras=args.use_all_cameras,
											  num_samples=args.max_train_samples,
											  num_train_samples=args.samples_from_train,
											  num_val_samples=args.num_val_samples)

		if json_dict is None:
			print("Error converting scene: {}, skipping...".format(scene))
			continue

		stats_dict["scene_name"].append(scene)
		stats_dict["num_train_views"].append(len(json_dict['frames']))
		stats_dict["num_bboxes"].append(len(json_dict['bounding_boxes']))
		stats_dict["width"].append(json_dict['w'])
		stats_dict["height"].append(json_dict['h'])
		stats_dict["trained"].append(True if scene in scenes_trained else False)

	stats_df = pd.DataFrame(stats_dict)
	stats_df.to_csv(os.path.join(args.output_path, "stats.csv"), index=False)

	print("Training and rendering...")
	for idx, scene in enumerate(scenes_needs_training):
		print("Training scene {}/{}: {}".format(idx+1, len(scenes_needs_training), scene))

		save_snapshot = ''
		if args.save_snapshot:
			save_snapshot = os.path.join(args.output_path, scene, 'train', args.save_snapshot)

		arg_list = [
			"python", str(args.run_script_path), 
			"--scene", str(os.path.join(args.output_path, scene, 'train')),
			"--mode", "nerf",
			"--save_snapshot", str(save_snapshot),
			"--screenshot_transforms", str(os.path.join(args.output_path, scene, 'val', 'val_transforms.json')),
			"--screenshot_dir", str(os.path.join(args.output_path, scene, 'val', 'screenshots')),
			"--screenshot_spp", str(args.screenshot_spp),
			"--n_steps", str(args.n_steps),
			"--width", str(int(stats_dict["width"][idx])),
			"--height", str(int(stats_dict["height"][idx])),
			"--save_log", str(os.path.join(args.output_path, scene, 'train', 'log')),
			"--show_bbox",
		]

		subprocess.run(arg_list)

		stats_dict["trained"][idx] = True
		stats_df = pd.DataFrame(stats_dict)
		stats_df.to_csv(os.path.join(args.output_path, "stats.csv"), index=False)

	print("Stats saved to {}".format(os.path.join(args.output_path, "stats.csv")))
	stats_df = pd.DataFrame(stats_dict)
	stats_df.to_csv(os.path.join(args.output_path, "stats.csv"), index=False)
