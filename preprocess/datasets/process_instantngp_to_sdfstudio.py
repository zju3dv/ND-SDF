## Copy from https://github.com/autonomousvision/sdfstudio/blob/master/scripts/datasets/process_nerfstudio_to_sdfstudio.py
import argparse
import json
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def main(args):
    """
    Given data that follows the nerfstudio format such as the output from colmap or polycam,
    convert to a format that sdfstudio will ingest
    """
    output_dir = Path(args.output_dir)
    data = Path(args.data)
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_params = json.load(open(data))
    
    # === load camera intrinsics and poses ===
    cam_intrinsics = []
    if args.data_type == "colmap":
        cam_intrinsics.append(np.array([
            [cam_params["fl_x"], 0, cam_params["cx"]],
            [0, cam_params["fl_y"], cam_params["cy"]],
            [0, 0, 1]]))
    R, totp = cam_params["R"], cam_params["totp"]
    avglen = cam_params["avglen"]
    frames = cam_params["frames"]
    poses = []
    image_paths = []
    # only load images with corresponding pose info
    # currently in random order??, probably need to sort
    for frame in frames:
        # load intrinsics from polycam
        if args.data_type == "polycam":
            cam_intrinsics.append(np.array([
                [frame["fl_x"], 0, frame["cx"]],
                [0, frame["fl_y"], frame["cy"]],
                [0, 0, 1]]))

        # load poses
        # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
        # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
        # IGNORED for now
        c2w = np.array(frame["transform_matrix"]).reshape(4, 4)
        c2w[0:3, 1:3] *= -1
        poses.append(c2w)

        # load images
        file_path = Path(frame["file_path"])
        img_path = file_path
        assert img_path.exists()
        image_paths.append(img_path)


    # Check correctness
    assert len(poses) == len(image_paths)
    assert len(poses) == len(cam_intrinsics) or len(cam_intrinsics) == 1

    # Filter invalid poses
    poses = np.array(poses)
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
    # === Normalize the scene ===
    scene_type = "indoor"
    if scene_type in ["indoor", "object"]:
        # Enlarge bbox by 1.05 for object scene and by 5.0 for indoor scene
        # TODO: Adaptively estimate `scene_scale_mult` based on depth-map or point-cloud prior
        if not args.scene_scale:
            args.scene_scale = 1.05 if scene_type == "object" else 5.0
        scene_scale = 2.0 / (np.max(max_vertices - min_vertices) + args.scene_scale)
        scene_center = (min_vertices + max_vertices) / 2.0
        # normalize pose to unit cube
        poses[:, :3, 3] -= scene_center
        poses[:, :3, 3] *= scene_scale
        # calculate scale matrix
        scale_mat = np.eye(4).astype(np.float32)
        scale_mat[:3, 3] -= scene_center
        scale_mat[:3] *= scene_scale
        scale_mat = np.linalg.inv(scale_mat)
    else:
        scene_scale = 1.0
        scale_mat = np.eye(4).astype(np.float32)
    print(scale_mat[0, 0], scene_center, np.max(max_vertices - min_vertices))
    # === Construct the scene box ===
    if scene_type == "indoor":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.01,
            "far": 2.5,
            "radius": 1.0,
            "collider_type": "box",
        }
    elif scene_type == "object":
        scene_box = {
            "aabb": [[-1, -1, -1], [1, 1, 1]],
            "near": 0.05,
            "far": 2.0,
            "radius": 1.0,
            "collider_type": "near_far",
        }
    elif scene_type == "unbound":
        # TODO: case-by-case near far based on depth prior
        #  such as colmap sparse points or sensor depths
        scene_box = {
            "aabb": [min_vertices.tolist(), max_vertices.tolist()],
            "near": 0.01,
            "far": 2.5 * np.max(max_vertices - min_vertices),
            "radius": np.min(max_vertices - min_vertices) / 2.0,
            "collider_type": "box",
        }

    sample_img = cv2.imread(str(image_paths[0]))
    h, w, _ = sample_img.shape
    tar_h, tar_w = h, w

    # === Construct the frames in the meta_data.json ===
    frames = []
    out_index = 0
    for idx, (valid, pose, image_path) in enumerate(tqdm(zip(valid_poses, poses, image_paths))):
        if not valid:
            continue
        # save rgb image path
        rgb_path = str(image_path.relative_to(output_dir))

        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": cam_intrinsics[0].tolist() if args.data_type == "colmap" else cam_intrinsics[idx].tolist()
        }
        img_name = os.path.basename(rgb_path).split('.')[0]
        out_depth_path = output_dir / "depth" / f"{img_name}.npy"
        out_normal_path = output_dir / "normal" / f"{img_name}.npy"

        depth_path = str(out_depth_path.relative_to(output_dir))
        normal_path = str(out_normal_path.relative_to(output_dir))

        frame["mono_depth_path"] = depth_path
        frame["mono_normal_path"] = normal_path

        frames.append(frame)
        out_index += 1

    # === Construct and export the metadata ===
    meta_data = {
        "camera_model": "OPENCV",
        "height": tar_h,
        "width": tar_w,
        "has_mono_prior": True,
        "has_sensor_depth": False,
        "has_foreground_mask": False,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
        "R": R,
        "totp": totp,
        "avglen": avglen,
        "frames": frames,
    }
    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    # === Generate mono priors using omnidata ===


    print(f"Done! The processed data has been saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess nerfstudio dataset to sdfstudio dataset, "
                                                 "currently support colmap and polycam")

    parser.add_argument("--data", required=True, help="path to json file")
    parser.add_argument("--output-dir", dest="output_dir", required=True, help="path to output data directory")
    parser.add_argument("--data-type", dest="data_type", required=True, choices=["colmap", "polycam"])
    parser.add_argument("--scene-scale", dest="scene_scale", type=float, default=None,
                        help="The bounding box of the scene is firstly calculated by the camera positions, "
                             "then add with scene_scale")
    args = parser.parse_args()

    main(args)
