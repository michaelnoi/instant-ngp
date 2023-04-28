import os
import torch
import numpy as np
import argparse
from functools import partial
from tqdm.contrib.concurrent import process_map


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def interpolate_features(filename, features_dir, output_dir, target_max_res):
    with np.load(os.path.join(features_dir, filename)) as f:
        rgbsigma = f['rgbsigma']
        res = f['resolution']

        # Convert density to alpha
        # rgbsigma[:, 3] = density_to_alpha(rgbsigma[:, 3])

        # First reshape from (H * W * D, C) to (D, H, W, C)
        rgbsigma = rgbsigma.reshape(res[2], res[1], res[0], -1)

        # Transpose to (C, W, H, D)
        rgbsigma = np.transpose(rgbsigma, (3, 2, 1, 0))
        rgbsigma = np.expand_dims(rgbsigma, axis=0)
        tensor = torch.from_numpy(rgbsigma)

        downsample_factor = target_max_res / res.max()
        tensor = torch.nn.functional.interpolate(tensor, scale_factor=downsample_factor, mode='nearest-exact')
        rgbsigma_new = tensor.numpy().squeeze()
        target_res = np.array(rgbsigma_new.shape[1:])

        print(target_res)

        # Back to (D, H, W, C)
        rgbsigma_new = np.transpose(rgbsigma_new, (3, 2, 1, 0))
        rgbsigma_new = rgbsigma_new.reshape(target_res[0] * target_res[1] * target_res[2], -1)

        np.savez_compressed(os.path.join(output_dir, filename),
                            rgbsigma=rgbsigma_new,
                            resolution=target_res,
                            bbox_min=f['bbox_min'], 
                            bbox_max=f['bbox_max'],
                            scale=f['scale'], 
                            offset=f['offset'],
                            from_mitsuba=f['from_mitsuba'],
                            angle=f['angle'])


def downsample_features(filename, features_dir, output_dir, downscale):
    avg_pool = torch.nn.AvgPool3d(downscale, ceil_mode=True)
    # max_pool = torch.nn.MaxPool3d(downscale, return_indices=True, ceil_mode=True)

    with np.load(os.path.join(features_dir, filename)) as f:
        rgbsigma = f['rgbsigma']
        res = f['resolution']

        # Convert density to alpha
        rgbsigma[:, 3] = density_to_alpha(rgbsigma[:, 3])

        # First reshape from (H * W * D, C) to (D, H, W, C)
        rgbsigma = rgbsigma.reshape(res[2], res[1], res[0], -1)

        # Transpose to (C, W, H, D)
        rgbsigma = np.transpose(rgbsigma, (3, 2, 1, 0))
        rgbsigma = np.expand_dims(rgbsigma, axis=0)
        tensor = torch.from_numpy(rgbsigma)

        tensor_pooled = avg_pool(tensor)
        # alpha_pooled, indices = max_pool(tensor[:, -1, ...])
        # tensor_pooled = tensor.flatten(2)[:, :, indices].squeeze(2)

        new_shape = tensor_pooled.shape[2:]
        # print(new_shape)
        assert (np.ceil(res / downscale) == new_shape).all()

        rgbsigma_pooled = tensor_pooled.numpy()
        # print(rgbsigma_pooled)

        rgbsigma_pooled = rgbsigma_pooled.squeeze()

        # Back to (D, H, W, C)
        rgbsigma_pooled = np.transpose(rgbsigma_pooled, (3, 2, 1, 0))
        rgbsigma_pooled = rgbsigma_pooled.reshape(new_shape[0] * new_shape[1] * new_shape[2], -1)

        # rgbsigma_pooled *= 255
        # rgbsigma_pooled = rgbsigma_pooled.astype(np.uint8)

        np.savez_compressed(os.path.join(output_dir, filename),
                            rgbsigma=rgbsigma_pooled,
                            resolution=np.ceil(res / downscale).astype(res.dtype),
                            bbox_min=f['bbox_min'], 
                            bbox_max=f['bbox_max'],
                            scale=f['scale'], 
                            offset=f['offset'],
                            from_mitsuba=f['from_mitsuba'])


def parse_args():
    parser = argparse.ArgumentParser(description='Downsample features.')

    parser.add_argument('--features_dir', '-f', type=str, help='Path to the features directory.')
    parser.add_argument('--output_dir', '-o', type=str, help='Path to the output directory.')
    parser.add_argument('--downscale', '-d', type=int, default=8, help='Downscale factor, used in pooling.')
    parser.add_argument('--interpolate', '-i', action='store_true', help='Interpolate features to target resolution.')
    parser.add_argument('--target_max_res', '-t', type=int, default=160, 
                        help='Target maximum resolution, used in interpolation.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    feat_files = [f for f in os.listdir(args.features_dir) if f.endswith('.npz')]

    if args.interpolate:
        fn = partial(interpolate_features, features_dir=args.features_dir, output_dir=args.output_dir, 
                     target_max_res=args.target_max_res)
    else:
        fn = partial(downsample_features, features_dir=args.features_dir, output_dir=args.output_dir, 
                     downscale=args.downscale)

    process_map(fn, feat_files, max_workers=16)
                            