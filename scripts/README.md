# instant-ngp Fork for NeRF Feature Extraction

This repo contains an instant-ngp fork modified to add uniform radiance and density sampling and object bounding box visualization. It is intended for the dataset creation of [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN), an 3D object detection framework on NeRF.

The fork is helpful if you plan to extract NeRF features from instant-ngp trained NeRF models, either on the [Hypersim](https://github.com/apple/ml-hypersim) and [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) we use, or on your custom scenes/datasets.

## Installation
Please follow the instructions [here](../README.md) for setup. You may also refer to the original [repo](https://github.com/NVlabs/instant-ngp), although some content might no longer apply after updates.


## Hypersim NeRF Training
We provide scripts facilitating the NeRF training for these two datasets.

For Hypersim, make sure you have cloned the [repo](https://github.com/apple/ml-hypersim) and downloaded the scene data needed. Only tone-mapped preview RGB images and metadata for each scene is needed for NeRF training, which can be downloaded using the handy [script](https://github.com/apple/ml-hypersim/tree/main/contrib/99991) from the Hypersim repo.

When the Hypersim data is ready, running `run_hypersim.py` automatically creates the file structure and `transforms.json` needed for instant-ngp NeRF training for every scene in the Hypersim data folder, and starts the training after that. 

Example usage:
```bash
python run_hypersim.py \
--hypersim_path /path/to/downloaded/scenes \
--metadata_path /path/to/ml-hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv \
--run_script_path ./run.py \
--output_path ./results \
--n_steps 50000 
```

You may find a source image folder, `transforms.json`, validation rendering results, and NeRF models saved for each scene under the output directory.


## 3D-FRONT NeRF Training
For scene layout configuration, rendering, and dataset preparation, please refer to our forked [BlenderProc repo](https://github.com/hjk0918/BlenderProc).

Once the data is generated using the repo above, `run_3dfront.py` can be used to train the NeRF:

```bash
python run_3dfront.py \
--data_path /path/to/3dfront/scenes \
--run_script_path ./run.py \
--output_path ./results \
--n_steps 50000
```

## NeRF Feature Extraction
We modified part of the instant-ngp code to add uniform sampling of the radiance and density of the trained NeRF models. Most of the modification can be found in `src/testbed_nerf.cu`, `src/python_api.cu`, and `scripts/run.py`.

After the NeRF models have been optimized, run `prepare_rpn_input.py` for feature extraction. Example:

```bash
python prepare_rpn_input.py \
--dataset_type <hypersim or 3dfront> \
--dataset_path /path/to/dataset \
--model_dir /path/to/trained/nerfs \
--snapshot_name model.msgpack \
--output_dir ./results \
--max_res 160 \
--use_dynamic_res
```

`--max_res` specify the resolution of the largest dimension of the scene, while the other two dimensions are scaled accordingly. The boundary of the scene can be estimated by the camera positions, object bounding boxes, or specified as input.

When the script completes, you will find `.npz` files under the output directory storing the radiance and density grid, as well as other instant-ngp scene data. The additional data is useful for future visualization. 

The default extraction method queries radiance from all input camera view directions and averages the results. It is also possible to sample from fixed directions or potentially visible camera directions. Instead of storing the radiance as RGB, code for approximating an SH at each sample point and saving the SH coefficients as features is also provided. You can switch between different methods by modifying the function invoked in the code.

**Note that the output RGB and density are in the shape of `(W * L * H, 4)`, and are under a y-up coordinate system. They need to be reshaped into `(W, L, H, 4)` and transposed to z-up before using as input for NeRF-RPN.**

A sample code for transformation:
```python
res = features['resolution']
rgbsigma = rgbsigma['rgbsigma']
rgbsigma = rgbsigma.reshape(res[2], res[1], res[0], -1) # to (H, L, W, 4)
rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)) # to (W, L, H, 4)
rgbsigma = np.transpose(rgbsigma, (2, 0, 1, 3)) # y-up to z-up, xyz -> yzx
```

## Bounding Box Visualization
To facilitate NeRF detection dataset cleaning and proposal visualization, we add bounding box visualization to the instant-ngp GUI. To enable it, add extra bounding box information in the `transforms.json` like this:

```json
"bounding_boxes": [
    {
        "extents": [2.7286, 2.096, 0.9446],
        "orientation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "position": [2.5532, -1.0173, 0.4723]
    },
    ...
]
```

Orientation is the rotation matrix applied around z axis. The other two properties also assume z axis is the vertical axis.

Then, in instant-ngp GUI, you will find a `Visualize object bounding boxes` checkbox under `Debug visualization` for visualizing the boxes.

