#!/usr/bin/env python3
"""
visualize_single_scene.py

Load a Contact-GraspNet .npz (result after inference) and render it with Mayavi using your existing helper functions.
"""

import os
import argparse
import numpy as np

from visualization_utils import visualize_grasps, show_image

def main():
    parser = argparse.ArgumentParser(description="Visualize a saved Contact-GraspNet .npz without re-running inference.")
    parser.add_argument("npz_path", help="Path to .npz file produced by Contact-GraspNet")
    args = parser.parse_args()

    # load npz file
    data = np.load(args.npz_path, allow_pickle=True)
    pred_grasps_cam = data["pred_grasps_cam"].item()
    scores          = data["scores"].item()
    contact_pts     = data["contact_pts"]
    pc_full         = data["pc_full"]
    pc_colors       = data["pc_colors"]
    segmap          = data["segmap"]
    rgb             = data["rgb"]  
    cam_K           = data["cam_K"]
    
    show_image(rgb, segmap)
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

if __name__ == "__main__":
    main()
