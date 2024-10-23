#!/usr/bin/env python3
import csv
import json
import os

import cv2
import numpy as np
import open3d as o3d
import rospy
import tf
import utils
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from open3d_ros_helper import open3d_ros_helper as orh
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Empty, EmptyResponse

from panoptic_mapping_msgs.msg import (
    DetectronLabel,
    DetectronLabels,
    KittiLabel,
    KittiLabels,
)


class KittiLidarCameraDataPlayer:

    def __init__(self):
        """Initialize ros node and read params"""
        # params
        self.data_path = rospy.get_param("~data_path")
        self.global_frame_name = rospy.get_param("~global_frame_name", "world")
        # TODO(thuaj): set this
        self.sensor_frame_name = rospy.get_param("~sensor_frame_name", "velodyne")
        self.use_detectron = rospy.get_param("~use_detectron", True)
        self.use_kitti_label = rospy.get_param("~use_kitti_label", False)
        self.play_rate = rospy.get_param("~play_rate", 1.0)
        self.wait = rospy.get_param("~wait", True)
        self.max_frames = rospy.get_param("~max_frames", 1e9)
        self.refresh_rate = 100  # Hz

        # ROS
        self.color_pub = rospy.Publisher("~color_image", Image, queue_size=100)
        self.id_pub = rospy.Publisher("~id_image", Image, queue_size=100)
        self.lidar_pub = rospy.Publisher("~lidar", PointCloud2, queue_size=100)
        if self.use_detectron:
            self.label_pub = rospy.Publisher("~labels", DetectronLabels, queue_size=100)
        if self.use_kitti_label:
            self.kitti_label_pub = rospy.Publisher("~kitti_labels", KittiLabels, queue_size=100)
        self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # setup
        self.cv_bridge = CvBridge()
        stamps_file = os.path.join(self.data_path, "timestamps.csv")
        self.times = []
        self.ids = []
        self.current_index = 0  # Used to iterate through
        if not os.path.isfile(stamps_file):
            rospy.logfatal("No timestamp file '%s' found." % stamps_file)
        with open(stamps_file) as read_obj:
            csv_reader = csv.reader(read_obj)
            for row in csv_reader:
                if row[0] == "ImageID":
                    continue
                self.ids.append(str(row[0]))
                self.times.append(float(row[1]))

        self.ids = [x for _, x in sorted(zip(self.times, self.ids))]
        self.times = sorted(self.times)
        self.times = [(x - self.times[0]) / self.play_rate for x in self.times]
        self.start_time = None

        if self.wait:
            self.start_srv = rospy.Service("~start", Empty, self.start)
        else:
            self.start(None)

    def start(self, _):
        self.running = True
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.refresh_rate), self.callback)
        return EmptyResponse()

    def callback(self, _):
        # Check we should be publishing.
        if not self.running:
            return

        # Check we're not done.
        if self.current_index >= len(self.times):
            rospy.loginfo("Finished playing the dataset.")
            rospy.signal_shutdown("Finished playing the dataset.")
            return

        # Check the time.
        now = rospy.Time.now()
        if self.start_time is None:
            self.start_time = now
        if self.times[self.current_index] > (now - self.start_time).to_sec():
            # print(self.times[self.current_index], (now - self.start_time).to_sec())
            return

        # Get all data and publish.
        color_file_folder = os.path.join(self.data_path, "image_2")
        color_file_id = os.path.join(color_file_folder, self.ids[self.current_index])
        pose_file_folder = os.path.join(self.data_path, "pose")
        pose_file_id = os.path.join(pose_file_folder, self.ids[self.current_index])
        lidar_file_folder = os.path.join(self.data_path, "pcd")
        lidar_file_id = os.path.join(lidar_file_folder, self.ids[self.current_index])
        kitti_label_file_foler = os.path.join(self.data_path, "labels")
        kitti_label_file_id = os.path.join(kitti_label_file_foler, self.ids[self.current_index])
        # Color.
        color_file = color_file_id + ".png"
        pose_file = pose_file_id + ".txt"
        lidar_file = lidar_file_id + ".pcd"
        files = [color_file, pose_file, lidar_file]
        if self.use_detectron:
            pred_file = color_file_id + "_predicted2.png"
            labels_file = color_file_id + "_labels.json"
            files += [pred_file, labels_file]
        if self.use_kitti_label:
            kitti_label_file = kitti_label_file_id + ".label"
            files += [kitti_label_file]

        for f in files:
            if not os.path.isfile(f):
                rospy.logwarn("Could not find file '%s', skipping frame." % f)
                self.current_index += 1
                return

        # Load and publish Color image.
        cv_img = cv2.imread(color_file)
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, "bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.sensor_frame_name
        self.color_pub.publish(img_msg)

        # Load and publish ID image.
        if self.use_detectron:
            cv_img = cv2.imread(pred_file)
            img_msg = self.cv_bridge.cv2_to_imgmsg(cv_img[:, :, 0], "8UC1")
            img_msg.header.stamp = now
            img_msg.header.frame_id = self.sensor_frame_name
            self.id_pub.publish(img_msg)

        # Load and publish labels.
        if self.use_detectron:
            # print("use detectron label")
            label_msg = DetectronLabels()
            label_msg.header.stamp = now
            with open(labels_file) as json_file:
                data = json.load(json_file)
                for d in data:
                    if "instance_id" not in d:
                        d["instance_id"] = 0
                    if "score" not in d:
                        d["score"] = 0
                    label = DetectronLabel()
                    label.id = d["id"]
                    label.instance_id = d["instance_id"]
                    label.is_thing = d["isthing"]
                    label.category_id = d["category_id"]
                    label.score = d["score"]
                    label_msg.labels.append(label)
            self.label_pub.publish(label_msg)

        # load and publish lidar pcd
        o3dpc = o3d.io.read_point_cloud(lidar_file)

        rospc = orh.o3dpc_to_rospc(o3dpc, frame_id=self.sensor_frame_name, stamp=now)
        self.lidar_pub.publish(rospc)

        # Load and publish transform.
        if os.path.isfile(pose_file):
            # pose_data = [float(x) for x in open(pose_file).read().split()]
            with open(pose_file) as f:
                x = f.read().split()
                pose_data = [float(i) for i in x]
            transform = np.eye(4)
            for row in range(4):
                for col in range(4):
                    transform[row, col] = pose_data[row * 4 + col]
            rotation = tf.transformations.quaternion_from_matrix(transform)
            self.tf_broadcaster.sendTransform(
                (transform[0, 3], transform[1, 3], transform[2, 3]),
                rotation,
                now,
                self.sensor_frame_name,
                self.global_frame_name,
            )
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.global_frame_name
        pose_msg.pose.position.x = pose_data[3]
        pose_msg.pose.position.y = pose_data[7]
        pose_msg.pose.position.z = pose_data[11]
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]
        self.pose_pub.publish(pose_msg)

        if self.use_kitti_label:
            kitti_label_msg = KittiLabels()
            kitti_label_msg.header.stamp = now
            labelscan = (np.fromfile(kitti_label_file, dtype=np.int32)).reshape(-1, 1)
            labeldata = utils.LabelDataConverter(labelscan)
            for i in range(len(labeldata.full_label)):
                kitti_label = KittiLabel()
                kitti_label.full_id = labeldata.full_label[i]
                kitti_label.semantic_id = labeldata.semantic_id[i]
                kitti_label.instance_id = labeldata.instance_id[i]
                kitti_label_msg.labels.append(kitti_label)
            self.kitti_label_pub.publish(kitti_label_msg)

        self.current_index += 1
        if self.current_index > self.max_frames:
            rospy.signal_shutdown("Played reached max frames (%i)" % self.max_frames)


if __name__ == "__main__":
    rospy.init_node("kitti_lidar_camera_player")
    kitti_lidar_camera_player = KittiLidarCameraDataPlayer()
    rospy.spin()
