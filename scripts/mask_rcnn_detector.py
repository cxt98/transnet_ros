import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
import threading
import argparse
import os
import sys
from copy import deepcopy

transnet_src_path = os.path.dirname(os.path.realpath(__file__)) + '/../src'
print(transnet_src_path)
if transnet_src_path not in sys.path:
    sys.path.append(transnet_src_path)

from TransNet.network_utility.segmentation.mask_rcnn import build_model

import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import ros_numpy


class MaskRCNN_Detector():
    def __init__(self, args):

        rospy.init_node(args.node_name)

        self.args = args
        self.maskrcnn_model = build_model(config={'num_classes': 7})
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.state_dict = torch.load(args.maskrcnn_weight_path, map_location = self.device)
        self.maskrcnn_model.load_state_dict(self.state_dict['model_state_dict'])
        self.maskrcnn_model.to(self.device)
        self.maskrcnn_model.eval()

        rospy.loginfo('Mask R-CNN Network loaded')
        
        self.category = {
            'bottle': 1,
            'bowl': 2,
            'container': 3,
            'tableware': 4,
            'water_cup': 5,
            'wine_cup': 6            
        }
        self.category_inverse = [
            'background',
            'bottle',
            'bowl',
            'container',
            'tableware',
            'water_cup',
            'wine_cup'
        ]

        self._lock = threading.RLock()

        rgb_sub = Subscriber(args.rgb_channel, Image)
        depth_sub = Subscriber(args.depth_channel, Image)
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.5)
        ts.registerCallback(self.img_callback)
        self.image_pub = rospy.Publisher(args.output_vis_channel, Image, queue_size=10)
        self.det_l_pub = rospy.Publisher(args.output_meta_channel, Float32MultiArray, queue_size=10)
        self.det_m_pub = rospy.Publisher(args.output_mask_channel, Float32MultiArray, queue_size=10)

        self.idx = 0
        self.depth_unit = 1000
        self._rgb_image, self._depth_image = None, None
        self.list = None
        self.maskrcnn_output = None
        self.img_vis = None

        if args.alwaysrun_tag:
            rospy.loginfo(f'always run inference in {args.hz} Hz')
        else: # will only run when activated
            rospy.loginfo(f'Waiting activation from {args.user_activate_channel} to run inference')
            self.run_net_sub = rospy.Subscriber(args.user_activate_channel, Bool, self.inference, queue_size=1)
    
    def crop_img(self):
        with self._lock: # TODO: only work for 720*1280 input
            self._rgb_image = cv2.resize(self._rgb_image[:, 160:1120], (640, 480), interpolation=cv2.INTER_NEAREST)
            self._depth_image = cv2.resize(self._depth_image[:, 160:1120], (640, 480), interpolation=cv2.INTER_NEAREST)

    def img_callback(self, rgb, depth):
        with self._lock:
            self._rgb_image = ros_numpy.numpify(rgb)
            self._depth_image = ros_numpy.numpify(depth).astype(np.float64) / 1000 # TODO: check bug
            if self._rgb_image.shape[0] != 480:
                self.crop_img()

    def inference(self, msg): # msg is for user activated run
        if self._rgb_image is None or self._depth_image is None:
            rospy.logwarn('No rgbd image received')
            return
        
        with self._lock:
            with torch.no_grad():
                self.list = self.maskrcnn_model([F.to_tensor(self._rgb_image).to('cuda')])[0]
            boxes = self.list['boxes'].cpu().numpy() #[N, 4]
            labels = self.list['labels'].cpu().numpy() #[N]
            scores = self.list['scores'].cpu().numpy() #[N]
            masks = self.list['masks'].cpu().numpy() #[N, 1, 480, 640]
            wine_cup_idx = (labels == 6) # TODO: change to config
            self.maskrcnn_output = (labels[wine_cup_idx], scores[wine_cup_idx], boxes[wine_cup_idx], masks[wine_cup_idx])
            print(self.idx)        
            self.idx += 1
            rospy.loginfo(f'Detected {len(boxes)} objects')

    def publish(self):
        if self.args.visualize_tag:
            self.visualize()
        if self.args.savefile_tag:
            self.savefile()
        if self.maskrcnn_output is not None:
            labels, scores, boxes, masks = self.maskrcnn_output
            meta_nparray = np.hstack((labels[:, np.newaxis], scores[:, np.newaxis], boxes)).astype(np.float32)
            meta_msg = Float32MultiArray(data=meta_nparray.flatten())
            mask_msg = Float32MultiArray(data=masks.flatten())
            self.det_l_pub.publish(meta_msg)
            self.det_m_pub.publish(mask_msg)

    def visualize(self):
        if self.maskrcnn_output is not None:
            with self._lock:
                labels, scores, boxes, masks = self.maskrcnn_output
                img = deepcopy(self._rgb_image)
                for idx in range(len(boxes)):
                    if labels[idx] <= 0:
                        continue
                    x1, y1, x2, y2 = boxes[idx]
                    mask = masks[idx, 0]
                    color = np.asarray([255, 182, 193])
                    m = (mask > 0.5)
                    mask[m] = 1
                    mask = mask.astype('uint8')
                    mask = (mask > 0)
                    start_pt = tuple((int(x1), int(y1)))
                    end_pt = tuple((int(x2), int(y2)))
                    img = cv2.rectangle(img, start_pt, end_pt , (255,255,255), thickness=2)
                    img[:, :][mask] = color
                    img = np.ascontiguousarray(img, dtype=np.uint8)
                    text_xy = [0, 0]
                    text_xy[0] = int(max((x1 + x2) / 2 - 18, 0))
                    text_xy[1] = int(max(y1 - 18, 0))
                    name = self.category_inverse[int(labels[idx])]
                    img = cv2.putText(img, name, (text_xy[0],text_xy[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(0,255,0))
                self.img_vis = img
                self.image_pub.publish(ros_numpy.msgify(Image, img, encoding='rgb8'))

    def savefile(self):
        if self.maskrcnn_output is not None:
            with self._lock:
                os.system(f'mkdir -p {self.args.savefile_path}')
                labels = self.maskrcnn_output[0]
                bowl_mask = (labels == 2)
                water_cup_mask = (labels == 5)
                wine_cup_mask = (labels == 6)
                water_cup_list = {}
                wine_cup_list = {}
                bowl_list = {}
                
                for k in self.list.keys():
                    self.list[k] = self.list[k].cpu().detach()
                    water_cup_list[k] = self.list[k][water_cup_mask]
                    bowl_list[k] = self.list[k][bowl_mask]
                    wine_cup_list[k] = self.list[k][wine_cup_mask]
                
                torch.save(self.list, self.args.savefile_path + f'all_{self.idx}.pt')
                torch.save(water_cup_list, self.args.savefile_path + f'watercup_{self.idx}.pt')
                torch.save(bowl_list, self.args.savefile_path + f'bowl_{self.idx}.pt')
                torch.save(wine_cup_list, self.args.savefile_path + f'winecup_{self.idx}.pt') 

                cv2.imwrite(self.args.savefile_path + f'rgb_{self.idx}.png', self._rgb_image[:, :, ::-1])
                cv2.imwrite(self.args.savefile_path + f'depth_{self.idx}.png', (self.depth_unit * self._depth_image).astype(np.uint16))
                if self.img_vis is not None:
                    cv2.imwrite(self.args.savefile_path + f'maskrcnn_vis_{self.idx}.png', self.img_vis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_name', type=str, default='MaskRCNN_Detector')
    parser.add_argument('--visualize_tag', type=bool, default=True, help='when set to true, will draw 2D mask and publish to output_channel')
    parser.add_argument('--savefile_tag', type=bool, default=False, help='when set to true, will save all image and detected metadata to savefile_path')
    parser.add_argument('--savefile_path', type=str, default='data/output/')
    parser.add_argument('--alwaysrun_tag', type=bool, default=True, help='when set to true, will run in constant frequency --hz, otherwise will run once received activation signal from --user_activate_channel')
    parser.add_argument('--hz', type=float, default=0.5)

    parser.add_argument('--maskrcnn_weight_path', type=str, default='src/transnet_ros/src/model/MaskRCNN/mask_rcnn_2.pt')
    parser.add_argument('--rgb_channel', type=str, default='/camera/color/image_raw')
    parser.add_argument('--depth_channel', type=str, default='/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--caminfo_channel', type=str, default='/camera/color/camera_info')
    parser.add_argument('--output_vis_channel', type=str, default='/maskrcnn_vis')
    parser.add_argument('--output_meta_channel', type=str, default='/maskrcnn_out_meta')
    parser.add_argument('--output_mask_channel', type=str, default='/maskrcnn_out_mask')
    parser.add_argument('--user_activate_channel', type=str, default='/maskrcnn_activate')


    args = parser.parse_args()
    
    node = MaskRCNN_Detector(args)

    r = rospy.Rate(args.hz)

    while not rospy.is_shutdown():
        if args.alwaysrun_tag:
            node.inference(None)
            r.sleep()
        node.publish()



