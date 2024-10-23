import csv
import json
import os
import shutil
import time
from multiprocessing import Pool

import cv2
import fire
import numpy as np
import open3d as o3d
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from numpy.linalg import inv

"""
This script organizes kitti data into a standard format

source data organization:

├── source folder
│   ├── data_odometry_color
│   ├── data_odometry_labels
│   ├── data_odometry_poses
│   ├── data_odometry_velodyne
|   ├── data_odometry_calib

target data organization:

├── target folder (defaults as xxx/dataset/sequences)
│   ├── 00
│   │   ├── velodyne
│   │   |   ├── 000000.bin
│   |   ├── labels
│   │   |   ├── 000000.label
│   │   ├── image2
│   │   |   ├── 000000.png
|   |   |   ├── 000000_detectron2.png
│   │   ├── image3 (default we do not use this but also store it in this folder)
│   │   |   ├── 000000.png
│   ├── 01
│   ├── ...
│   ├── poses.txt
│   ├── calib.txt
│   ├── timestamps.txt

"""


def unzip_file(args):
    ziped_source_folder, unziped_target_folder = args
    # Ensure the target directory exists
    os.makedirs(unziped_target_folder, exist_ok=True)

    # Unzip the archive
    shutil.unpack_archive(ziped_source_folder + ".zip", unziped_target_folder)
    return f"Unzipped data from {ziped_source_folder}.zip to {unziped_target_folder}"


def unzip_multiple_archives(nproc, process_list, source_folder, target_folder):
    # Define the archives and their target folders
    archive_names = process_list

    # Prepare the argument list for multiprocessing
    args = [
        (os.path.join(source_folder, archive_name), os.path.join(target_folder, archive_name))
        for archive_name in archive_names
    ]

    with Pool(nproc) as pool:
        results = pool.imap_unordered(unzip_file, args)  # noqa


def move_single_sequence(args):
    process_list, seq, source_folder, target_folder = args
    for process_key in process_list:
        source_data_folder = os.path.join(source_folder, process_key)
        if process_key == "data_odometry_color":
            print(f"process color images {source_data_folder}")
            # for seq in sequence_list:
            seq_source_folder = os.path.join(source_data_folder, "dataset/sequences", seq, "image_2")
            seq_target_folder = os.path.join(target_folder, seq, "image_2")
            os.makedirs(seq_target_folder, exist_ok=True)
            # Move files from source to destination
            for file_name in os.listdir(seq_source_folder):
                source_file = os.path.join(seq_source_folder, file_name)
                destination_file = os.path.join(seq_target_folder, file_name)
                shutil.move(source_file, destination_file)
            # print(f"Moved data from {seq_source_folder} to {seq_target_folder}")
            seq_source_folder = os.path.join(source_data_folder, "dataset/sequences", seq, "image_3")
            seq_target_folder = os.path.join(target_folder, seq, "image_3")
            os.makedirs(seq_target_folder, exist_ok=True)
            # Move files from source to destination
            for file_name in os.listdir(seq_source_folder):
                source_file = os.path.join(seq_source_folder, file_name)
                destination_file = os.path.join(seq_target_folder, file_name)
                shutil.move(source_file, destination_file)
            # print(f"Moved data from {seq_source_folder} to {seq_target_folder}")
        if process_key == "data_odometry_labels":
            print(f"process semantic labels {source_data_folder}")
            # for seq in sequence_list:
            seq_source_folder = os.path.join(source_data_folder, "dataset/sequences", seq, "labels")
            seq_target_folder = os.path.join(target_folder, seq, "labels")
            os.makedirs(seq_target_folder, exist_ok=True)
            # Move files from source to destination
            for file_name in os.listdir(seq_source_folder):
                source_file = os.path.join(seq_source_folder, file_name)
                destination_file = os.path.join(seq_target_folder, file_name)
                shutil.move(source_file, destination_file)
            # print(f"Moved data from {seq_source_folder} to {seq_target_folder}")
        if process_key == "data_odometry_poses":
            print(f"process odometry poses {source_data_folder}")
            # for seq in sequence_list:
            seq_source_file = os.path.join(source_data_folder, "dataset/poses", f"{seq}.txt")
            seq_target_folder = os.path.join(target_folder, seq)
            seq_target_file = os.path.join(target_folder, seq, "poses.txt")
            os.makedirs(seq_target_folder, exist_ok=True)
            # Move files from source to destination
            shutil.move(seq_source_file, seq_target_file)
            # print(f"Moved data from {seq_source_file} to {seq_target_file}")
        if process_key == "data_odometry_velodyne":
            print(f"process lidar velodyne {source_data_folder}")
            # for seq in sequence_list:
            seq_source_folder = os.path.join(source_data_folder, "dataset/sequences", seq, "velodyne")
            seq_target_folder = os.path.join(target_folder, seq, "velodyne")
            os.makedirs(seq_target_folder, exist_ok=True)
            # Move files from source to destination
            for file_name in os.listdir(seq_source_folder):
                source_file = os.path.join(seq_source_folder, file_name)
                destination_file = os.path.join(seq_target_folder, file_name)
                shutil.move(source_file, destination_file)
            # print(f"Moved data from {seq_source_folder} to {seq_target_folder}")
        if process_key == "data_odometry_calib":
            print(f"process data_odometry_calib {source_data_folder}")
            # for seq in sequence_list:
            seq_source_folder = os.path.join(source_data_folder, "dataset/sequences", seq)
            seq_target_folder = os.path.join(target_folder, seq)
            os.makedirs(seq_target_folder, exist_ok=True)
            # Move files from source to destination
            for file_name in os.listdir(seq_source_folder):
                source_file = os.path.join(seq_source_folder, file_name)
                destination_file = os.path.join(seq_target_folder, file_name)
                shutil.move(source_file, destination_file)
            # print(f"Moved data from {seq_source_folder} to {seq_target_folder}")


def move_files(nproc, process_list, source_folder, target_folder, sequence_list):
    args_lists = [(process_list, seq, source_folder, target_folder) for seq in sequence_list]

    with Pool(nproc) as pool:
        results = pool.imap_unordered(move_single_sequence, args_lists)  # noqa


def create_labels(meta_data, output_file: str = ""):
    sizes = [
        'L', 'M', 'L', 'M', 'L', 'L', 'L', 'L', 'L', 'M', 'M', 'M', 'S', 'L',
        'S', 'M', 'M', 'L', 'M', 'L', 'L', 'L', 'L', 'L', 'M', 'S', 'S', 'S',
        'S', 'S', 'M', 'M', 'S', 'M', 'M', 'S', 'S', 'M', 'S', 'S', 'S', 'S',
        'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S',
        'M', 'L', 'M', 'L', 'M', 'M', 'M', 'S', 'S', 'S', 'S', 'S', 'M', 'M',
        'S', 'M', 'L', 'S', 'M', 'M', 'S', 'M', 'S', 'S'
    ]  # fmt: skip
    if output_file:  # noqa
        with open(output_file, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["InstanceID", "ClassID", "PanopticID", "Name", "Size"])
            writer.writerow([0, 0, 0, "Unknown", "M"])
            id = 1  # noqa
            for label in meta_data.stuff_classes:
                writer.writerow([id, id, 0, label, "L"])
                id += 1
            for i, label in enumerate(meta_data.thing_classes):
                writer.writerow([id, id, 1, label, sizes[i]])
                id += 1
        return len(meta_data.stuff_classes), "Saved %i labels in '%s'." % (id, output_file)
    else:
        return len(meta_data.stuff_classes), ""


def create_predictions(source_path, output_label_file, model):
    # Verify.
    if not os.path.isdir(source_path):
        print("Error: Directory '%s' does not exist." % source_path)
        return
    print("Processing target '%s'." % source_path)

    # Setup model.
    print("Setting up Detectron2 model... ", end="", flush="True")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)
    print("done!")

    # Setup labels.
    print("Setting up labels... ", end="", flush="True")
    meta_data = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    label_offset, msg = create_labels(meta_data, output_label_file)
    print("done!")
    if msg:
        print(msg)

    # Get files to parse.
    files = [o for o in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, o))]
    files = [f for f in files if f.endswith(".png")]
    times = []

    # Run inference.
    msg = "Predicting %i images... " % len(files)
    for i, im_file in enumerate(files):
        print(msg + f"{i / len(files) * 100:.1f}%", end="\r", flush=True)
        im = cv2.imread(os.path.join(source_path, im_file))
        # Predict.
        t1 = time.perf_counter()
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        t2 = time.perf_counter()
        times.append(t2 - t1)

        # Write output.
        file_id = im_file[:6]
        id_img = panoptic_seg.cpu().numpy()
        cv2.imwrite(os.path.join(source_path, file_id + "_predicted2.png"), id_img)

        for segment_info in segments_info:
            if segment_info["isthing"]:
                segment_info["category_id"] += label_offset
            segment_info["category_id"] += 1  # Compensate for unknown class.
        with open(os.path.join(source_path, file_id + "_labels.json"), "w") as json_file:
            json.dump(segments_info, json_file)
    print(msg + "done!")

    # Finish.
    times = np.array(times, dtype=float) * 1000
    print(f"Average inference time was {np.mean(times):.1f} +/- {np.std(times):.1f} ms per frame.")
    print("Finished parsing '%s'." % source_path)


# Folder 'poses':

# The folder 'poses' contains the ground truth poses (trajectory) for the
# first 11 sequences. This information can be used for training/tuning your
# method. Each file xx.txt contains a N x 12 table, where N is the number of
# frames of this sequence. Row i represents the i'th pose of the left camera
# coordinate system (i.e., z pointing forwards) via a 3x4 transformation
# matrix. The matrices are stored in row aligned order (the first entries
# correspond to the first row), and take a point in the i'th coordinate
# system and project it into the first (=0th) coordinate system. Hence, the
# translational part (3x1 vector of column 4) corresponds to the pose of the
# left camera coordinate system in the i'th frame with respect to the first
# (=0th) frame. Your submission results must be provided using the same data
# format.


def read_calib_file(filename):
    """read calibration file

    returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)  # noqa
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose

    calib_file.close()

    print("calib: ", calib)
    return calib


def process_poses_single(args):
    source_folder, seq = args
    # Load the KITTI pose file
    # using the ground truth provided by kitti visual odometry
    # NOTE(thuaj): semantic kitti also provided a gt pose generate by suma and it is called poses.txt
    # poses = np.loadtxt(os.path.join('/home/hts/Data/dataset/KITTI/dataset/sequences',seq, 'poses.txt'))
    poses_file_path = os.path.join(source_folder, seq, "poses.txt")
    new_poses_folder = os.path.join(source_folder, seq, "pose")

    calib_file_path = os.path.join(source_folder, seq, "calib.txt")

    calibration = read_calib_file(calib_file_path)

    # Create a folder to save the new pose text files
    if not os.path.exists(new_poses_folder):
        os.makedirs(new_poses_folder)

    pose_file = open(poses_file_path)  # noqa

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    i = 0
    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        lidar_pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))  # lidar pose in world frame

        # Save the pose to a file with a zero-padded filename
        filename = f"{i:06d}.txt"
        filepath = os.path.join(new_poses_folder, filename)
        np.savetxt(filepath, lidar_pose, fmt="%.6e")
        i = i + 1


def process_poses(nproc, source_folder, sequence_list):
    args = [(source_folder, seq) for seq in sequence_list]
    with Pool(nproc) as pool:
        results = pool.imap_unordered(process_poses_single, args)  # noqa


def generate_time(source_folder, seq):

    txt_file_path = os.path.join(source_folder, seq, "times.txt")
    # Open the txt file
    with open(txt_file_path) as file:
        lines = file.readlines()

    # Convert the timestamp values
    timestamps = [float(line.strip()) for line in lines]

    # Generate the list of ImageIDs
    imageids = [str(i).zfill(6) for i in range(len(timestamps))]

    timestamp_path = os.path.join(source_folder, seq, "timestamps.csv")

    # Write the ImageIDs and timestamps to a csv file
    with open(timestamp_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageID", "TimeStamp"])
        for imageid, timestamp in zip(imageids, timestamps):
            writer.writerow([imageid, timestamp])


# read KITTI bin file
def read_kitti_bin(file):
    with open(file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(-1, 4)


# convert KITTI bin to PCD
def convert_kitti_to_pcd_single(args):
    input_folder, output_folder = args
    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # loop through all KITTI bin files in input folder
    for file in os.listdir(input_folder):
        if file.endswith(".bin"):
            # read KITTI bin file
            data = read_kitti_bin(os.path.join(input_folder, file))

            # create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(np.zeros((data.shape[0], 3)))

            # save PCD file
            o3d.io.write_point_cloud(os.path.join(output_folder, file.replace(".bin", ".pcd")), pcd)


def convert_kitti_to_pcd(data_folder, seq_list, nproc=1):
    args = []
    for seq in seq_list:
        input_folder = os.path.join(data_folder, seq, "velodyne")
        output_folder = os.path.join(data_folder, seq, "pcd")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        args.append((input_folder, output_folder))
    with Pool(nproc) as pool:
        results = pool.imap_unordered(convert_kitti_to_pcd_single, args)  # noqa


def process_main(
    source_folder,
    target_folder,
    process_list: list = [
        "data_odometry_velodyne",
        "data_odometry_color",
        "data_odometry_labels",
        "data_odometry_poses",
        "data_odometry_calib",
    ],
    sequence_list: list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
):
    unziped_source_folder = source_folder
    nproc = 4
    # >>> step 1 unzip and rearrange folder structure
    unzip_multiple_archives(nproc, process_list, source_folder, unziped_source_folder)
    print(f"kitti seq: {sequence_list}")
    move_files(nproc, process_list, unziped_source_folder, target_folder, sequence_list)
    # >>> step2 generate detectron labels
    model = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    for seq in sequence_list:
        image_source_path = os.path.join(target_folder, seq, "image_2")
        image_output_label_file = os.path.join(target_folder, seq, "detectron_labels.csv")
        create_predictions(image_source_path, image_output_label_file, model)
    # >>> step3 generate poses
    process_poses(nproc, target_folder, sequence_list)
    # >>> step4 generate timestamps
    for seq in sequence_list:
        generate_time(target_folder, seq)
    # >>> step5 convert KITTI bin to PCD
    convert_kitti_to_pcd(target_folder, sequence_list, nproc)


if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(process_main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")
