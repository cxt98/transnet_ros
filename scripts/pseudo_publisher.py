import rospy
import ros_numpy

from sensor_msgs.msg import Image, CameraInfo
import cv2
import os
import numpy as np
import argparse

# read image files to send out ros messages
class PsuedoPublisher:
    def __init__(self, args) -> None:
        
        rospy.init_node(args.node_name)
        self.args = args
        self.img_rgb_pub   = rospy.Publisher(args.pseudo_cam + args.rgb_channel, Image, queue_size=1)
        self.img_depth_pub = rospy.Publisher(args.pseudo_cam + args.depth_channel, Image, queue_size=1)
        self.caminfo_pub   = rospy.Publisher(args.pseudo_cam + args.caminfo_channel, CameraInfo, queue_size=1)
        # rgb-d image pairs from disk
        self.img_folder = args.folder
        if self.args.single_rgb:
            self.img_count = 1
        elif self.args.image_organized:
            self.img_count = args.img_count
        else:
            img_filenames = sorted(os.listdir(f'{self.img_folder}/rgb'))
            self.img_count = len(img_filenames)
            self.img_names = img_filenames
        self.img_ind = 0
        self.K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]])
        
    
    def publish_image(self):
        if self.args.single_rgb:
            img_rgb = cv2.imread(self.args.single_filename)
        elif self.args.image_organized:
            img_rgb = cv2.imread(f'{self.img_folder}/{self.img_ind:06d}-color.png')
            img_depth = cv2.imread(f'{self.img_folder}/{self.img_ind:06d}-depth.png', -1)
        else:
            img_rgb = cv2.imread(f'{self.img_folder}/rgb/{self.img_names[self.img_ind]}')
            img_depth = cv2.imread(f'{self.img_folder}/depth/{self.img_names[self.img_ind]}', -1)
        
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        self.img_rgb_pub.publish(ros_numpy.msgify(Image, img_rgb, encoding="rgb8"))
        if not self.args.single_rgb:
            self.img_depth_pub.publish(ros_numpy.msgify(Image, img_depth, encoding="16UC1"))
        caminfo = CameraInfo()
        caminfo.height, caminfo.width = 480, 640
        caminfo.K = self.K.flatten().tolist()
        P = np.hstack((self.K, np.array([[0., 0., 0.]]).T)) # image_geometry/pinhole_camera_model.h reads P to get fx, fy, cx, cy
        caminfo.P = P.flatten().tolist()
        # print(caminfo.K)
        self.caminfo_pub.publish(caminfo)

        self.img_ind = (self.img_ind + 1) % self.img_count
        if self.args.single_rgb:
            rospy.loginfo(f'published image %s', self.args.single_filename)
        else:
            rospy.loginfo(f'published image pair {self.img_ind} in folder %s', self.args.folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_name', type=str, default='Pseudo_Image_Publisher')
    parser.add_argument('--pseudo_cam', '-p', type=str, default='/camera')
    parser.add_argument('--rgb_channel', type=str, default='/color/image_raw')
    parser.add_argument('--depth_channel', type=str, default='/aligned_depth_to_color/image_raw')
    parser.add_argument('--caminfo_channel', type=str, default='/color/camera_info')
    parser.add_argument('--single_rgb', type=bool, default=False) # single file will override folder
    parser.add_argument('--single_filename', type=str, default='/home/cxt/Documents/others/ford/packing_box_perception/test_images_2cam/cam_1/frame0000.jpg')
    parser.add_argument('--folder', '-f', type=str, default='data/differentview_1/view2')
    parser.add_argument('--image_organized', type=bool, default=False) # for organized image, they are named as %06d-{color, depth}.png, otherwise, they have the same names in rgb/ and depth/ folders under --folder
    parser.add_argument('--img_count', '-n', type=int, default=100)
    parser.add_argument('--fx', type=float, default=601.333) # camera intrinsic of realsense L515 for clearpose dataset
    parser.add_argument('--fy', type=float, default=601.333)
    parser.add_argument('--cx', type=float, default=334.667)
    parser.add_argument('--cy', type=float, default=248.)
    args = parser.parse_args()
    p = PsuedoPublisher(args)
    while not rospy.is_shutdown():
        p.publish_image()
        rospy.sleep(1)