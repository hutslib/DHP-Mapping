"""
Author: thuaj@connect.ust.hk
Date: 2024-01-31 10:51:04
LastEditTime: 2024-10-23 12:39:39
Description: visualization detectron panoptic segmentation results
Copyright (c) 2024 by thuaj@connect.ust.hk, All Rights Reserved.
"""

import json
import os
from multiprocessing import Pool

import fire
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from PIL import Image

"""
usage example: python3 vis_detectron_segmentation.py \
    --color_file="colors-bright.txt" \
    --data_root_dir="~/Data/KITTI" \
    --nproc=4
"""


class Label:

    def __init__(self, instance_id, class_id, panoptic_id, r, g, b, name, size):
        self.instance_id = instance_id
        self.class_id = class_id
        self.panoptic_id = panoptic_id
        self.r = r
        self.g = g
        self.b = b
        self.name = name
        self.size = size


# Function to get color ID from JSON data
def get_color_id_from_json(pixel_value, json_data, colormap):
    if pixel_value == 0:  # noqa
        return colormap[0]
    else:
        idx = np.where(pixel_value == np.array([element["id"] for element in json_data]))[0]
        if idx.size > 0:
            class_id = json_data[idx[0]]["category_id"]
            return colormap[class_id]


def process(data_root_dir, seq, colormap):
    # 读取文件夹中的所有PNG文件
    folder_path = os.path.join(data_root_dir, seq, "image_2")
    save_folder_path = os.path.join(data_root_dir, seq, "vis_segmentation_2")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    file_list = os.listdir(folder_path)
    png_files = natsorted([f for f in file_list if f.endswith("_predicted2.png")])
    json_files = [f.replace("_predicted2.png", "_labels.json") for f in png_files]
    for png_file, json_file in zip(png_files, json_files):
        # 读取PNG文件并转换为numpy数组
        img = Image.open(os.path.join(folder_path, png_file))
        img_data = np.array(img)
        with open(os.path.join(folder_path, json_file)) as jf:
            json_data = json.load(jf)
        # Process image data using NumPy operations
        h, w = img_data.shape
        pixel_values = img_data.flatten()
        color_img_data_flat = np.array([get_color_id_from_json(value, json_data, colormap) for value in pixel_values])
        color_img_data = color_img_data_flat.reshape((h, w, 3))
        # 将numpy数组转换为PIL Image对象并保存为PNG文件
        color_img = Image.fromarray(color_img_data.astype(np.uint8))
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        color_img.save(os.path.join(save_folder_path, f"color_{png_file}"))
    print(f"finish {seq}")


def main(color_file, data_root_dir, nproc):

    rgb_values = []
    with open(color_file) as f:
        next(f)  # skip the first line
        for line in f:
            r, g, b = map(int, line.strip().split()[1:])
            rgb_values.append((r, g, b))
    #         print((r, g, b))

    colormap_normalized = [tuple(x / 255 for x in rgb) for rgb in rgb_values]
    colormap = [tuple(x for x in rgb) for rgb in rgb_values]

    num_per_row = 25
    num_rows = int(np.ceil(len(colormap) / num_per_row))

    fig, ax = plt.subplots(num_rows, num_per_row, figsize=(25, 2 * num_rows))
    ax = ax.flatten()

    for i, normalized_color in enumerate(colormap_normalized):
        rect = plt.Rectangle((0, 0), 1.0, 1.0, color=normalized_color)
        ax[i].add_artist(rect)
        ax[i].text(0.5, 0.1, str(i), ha="center", va="center", fontsize=18)
        ax[i].axis("off")

    for i in range(len(colormap_normalized), len(ax)):
        ax[i].axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(color_file.replace(".txt", "colormap.png"))

    #  setp 2 process each sequence
    seq_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    if nproc <= 1:
        for seq in seq_list:
            process(data_root_dir, seq, colormap)
    else:
        with Pool(processes=nproc) as p:
            p.starmap(process, [(data_root_dir, seq, colormap) for seq in seq_list])


if __name__ == "__main__":
    fire.Fire(main)
