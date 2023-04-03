from collections import defaultdict
import datetime
import random
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import time
from typing import Union

import numpy as np
import kornia
import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import SoccerPitch


from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate, split_circle_central
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz
from results_presentation import overlap_inout
import os
import cv2 as cv

def save_frames(video_path, output_dir, save=False):
    # Open the video file for reading
    video = cv.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize frame counter
    if  save:
        frame_count = 0
    else: 
        frames = []
        
    # Loop through all frames in the video
    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # If we reached the end of the video, exit the loop
        if not ret:
            break
        if save == True:
            # Save the current frame as a PNG image with an incremental filename
            filename = f"{frame_count:05d}.png"
            output_path = os.path.join(output_dir, filename)
            cv.imwrite(output_path, frame)
            # Increment the frame counter
            frame_count += 1
        else:
            frames.append(frame)

    # Release the video object and print a message
    video.release()
    if not save:
        return frames

def find_mp4_files(directory):
    mp4_files = []
    for file in os.listdir(directory):
        if file.endswith(".mp4"):
            mp4_files.append(file)
    return mp4_files

def convert_to_mp4(dir_path, output_file='output.mp4', fps=25.0, res_width=1120, res_height=540, extention='.png'):
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(dir_path, output_file)
    out = cv.VideoWriter(output_path, fourcc, fps, (res_width, res_height))

    nr_frames = len([f for f in os.listdir(dir_path) if f.endswith(extention)])
    # Iterate over frames and write them to video
    for i in range(1,nr_frames+1):
        filename = "{:05d}".format(i)+extention
        frame_path = os.path.join(dir_path, filename)
        frame = cv.imread(frame_path)
        print(filename, end='\r')
        out.write(frame)
    out.release()


args = Namespace(
        images_path=Path("data/datasets/demo/prova"),
        output_dir=Path("tmp/demo/prova"),
        checkpoint="data/segment_localization/train_59.pt",
        gpu=True,
        nworkers=16,
        batch_size_seg=16,
        batch_size_calib=256,
        image_width=1280,
        image_height=720,
        optim_steps=2000,
        lens_dist=False,
        write_masks=False,
        video_input=True,
        video_output=True
    )
device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

input_images = []
# Example usage:
if args.video_input:
    directory = args.images_path
    mp4_files = find_mp4_files(directory)
    for video in mp4_files:
        video_path = os.path.join(directory,video)
        #input_images = save_frames(video_path, args.output_dir)
        save_frames(video_path, directory, True)
        input_images = args.images_path
else: 
    input_images = args.images_path

object3d = SoccerPitchLineCircleSegments(
    device=device, base_field=SoccerPitchSNCircleCentralSplit()
)
object3dcpu = SoccerPitchLineCircleSegments(
    device="cpu", base_field=SoccerPitchSNCircleCentralSplit()
)

lines_palette = [0, 0, 0]
for line_class in SoccerPitch.lines_classes:
    lines_palette.extend(SoccerPitch.palette[line_class])

fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
fn_get_line_extremities = partial(get_line_extremities, maxdist=30, width=455, height=256, num_points_lines=4, num_points_circles=8)

dataset_seg = InferenceDatasetSegmentation(
    input_images, args.image_width, args.image_height
)
print("number of images:", len(dataset_seg))
dataloader_seg = torch.utils.data.DataLoader(
    dataset_seg,
    batch_size=args.batch_size_seg,
    num_workers=args.nworkers,
    shuffle=False,
    collate_fn=custom_list_collate,
)

model_seg = InferenceSegmentationModel(args.checkpoint, device)

image_ids = []
keypoints_raw = []
(args.output_dir / "masks").mkdir(parents=True, exist_ok=True)

#start segmentation
start_time = time.time()
for batch_dict in tqdm(dataloader_seg):
    # semantic segmentation
    # image_raw: [B, 3, image_height, image_width]
    # image: [B, 3, 256, 455]
    with torch.no_grad():
        sem_lines = model_seg.inference(batch_dict["image"].to(device))
    sem_lines = sem_lines.cpu().numpy().astype(np.uint8)  # [B, 256, 455]
    # point selection
    with Pool(args.nworkers) as p:
        skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
        keypoints_raw_batch = p.map(fn_get_line_extremities, skeletons_batch)

    # write to file
    if args.write_masks:
        #print("Write masks to file")
        for image_id, mask in zip(batch_dict["image_id"], sem_lines):
            mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
            mask.putpalette(lines_palette)
            mask.convert("RGB").save(args.output_dir / "masks" / image_id)

    image_ids.extend(batch_dict["image_id"])
    keypoints_raw.extend(keypoints_raw_batch)
seg_time = time.time() - start_time
print("segmentation_time --- %s seconds ---" % (seg_time))

#start calibration
model_calib = TVCalibModule(
    object3d,
    get_cam_distr(1.96, args.batch_size_calib, 1),
    get_dist_distr(args.batch_size_calib, 1) if args.lens_dist else None,
    (args.image_height, args.image_width),
    args.optim_steps,
    device,
    log_per_step=False,
    tqdm_kwqargs=None,
)


dataset_calib = InferenceDatasetCalibration(keypoints_raw, args.image_width, args.image_height, object3d)
dataloader_calib = torch.utils.data.DataLoader(dataset_calib, args.batch_size_calib, collate_fn=custom_list_collate)

per_sample_output = defaultdict(list)
per_sample_output["image_id"] = [[x] for x in image_ids]
for x_dict in dataloader_calib:


    _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

    points_line = x_dict["lines__px_projected_selection_shuffled"]
    points_circle = x_dict["circles__px_projected_selection_shuffled"]
    #print(f"{points_line.shape=}, {points_circle.shape=}")

    per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
    output_dict = tensor2list(detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss}))
    
    output_dict["points_line"] = points_line
    output_dict["points_circle"] =  points_circle
    for k in output_dict.keys():
        per_sample_output[k].extend(output_dict[k])

print("calib_time --- %s seconds ---" % (time.time() - start_time - seg_time))
print("tot_time --- %s seconds ---" % (time.time() - start_time))

df = pd.DataFrame.from_dict(per_sample_output)

df = df.explode(column=[k for k, v in per_sample_output.items() if isinstance(v, list)])
df.set_index("image_id", inplace=True, drop=False)

df.to_csv("csv_points.csv")

if args.video_output:
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

for sample in df.iloc:
    #sample = df.iloc[i]

    image_id = Path(sample.image_id).stem
    #print(f"{image_id=}")
    image = Image.open(input_images / sample.image_id).convert("RGB")
    image = T.functional.to_tensor(image)

    cam = get_camera_from_per_sample_output(sample, args.lens_dist)
    #print(cam, cam.str_lens_distortion_coeff(b=0) if args.lens_dist else "")
    points_line, points_circle = sample["points_line"], sample["points_circle"]

    if args.lens_dist:
        # we visualize annotated points and image after undistortion
        image = cam.undistort_images(image.unsqueeze(0).unsqueeze(0)).squeeze()
        # print(points_line.shape) # expected: (1, 1, 3, S, N)
        points_line = SNProjectiveCamera.static_undistort_points(points_line.unsqueeze(0).unsqueeze(0), cam).squeeze()
        points_circle = SNProjectiveCamera.static_undistort_points(points_circle.unsqueeze(0).unsqueeze(0), cam).squeeze()
    else:
        psi = None


    fig, ax = viz.init_figure(args.image_width, args.image_height)
    ax = viz.draw_image(ax, image)
    ax = viz.draw_reprojection(ax, object3dcpu, cam)
    ax = viz.draw_selected_points(
        ax,
        object3dcpu,
        points_line,
        points_circle,
        kwargs_outer={
            "zorder": 1000,
            "rasterized": False,
            "s": 500,
            "alpha": 0.3,
            "facecolor": "none",
            "linewidths": 3,
        },
        kwargs_inner={
            "zorder": 1000,
            "rasterized": False,
            "s": 50,
            "marker": ".",
            "color": "k",
            "linewidths": 4.0,
        },
    )
    dpi = 50
    plt.savefig(args.output_dir / f"{image_id}.png", dpi=dpi)
    plt.close()
    #plt.savefig(args.output_dir / f"{image_id}.pdf", dpi=dpi)
    #plt.savefig(args.output_dir / f"{image_id}.svg", dpi=dpi)

if args.video_output:
    convert_to_mp4(args.output_dir)