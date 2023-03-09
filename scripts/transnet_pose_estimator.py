import os
import sys
import numpy as np
import torch
import threading
import cv2
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from absl import app
import absl.flags as flags


transnet_src_path = os.path.dirname(os.path.realpath(__file__)) + '/../src'

if transnet_src_path not in sys.path:
    sys.path.append(transnet_src_path)

from TransNet.config.config import *
from TransNet.data.dataset import PoseEstimationDataset as PoseDataset
from TransNet.network_utility.network import TransNet
from baselines.GPVPose.tools.dataset_utils import get_2d_coord_np
from baselines.GPVPose.tools.eval_utils import get_bbox

import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import ros_numpy


class TransNet_Pose_Estimator():
    def __init__(self, args):

        rospy.init_node(args.node_name)

        self.args = args

        self.transnet_model = TransNet().to(args.device)
        self.transnet_model = self.transnet_model.eval()
        state_dict = torch.load(args.transnet_weight_path)
        self.transnet_model.posenet.load_state_dict(state_dict['posenet_state_dict'])
        rospy.loginfo('Finish initializing TransNet')
        self.dataset = PoseDataset(dataset_name="train", args=args, root=args.dataset_dir)
        self.output = None
        self.vis_img = None
        self.Tcam2robot = None
        self._rgb_image = None
        self._depth_image = None
        self.maskrcnn_output = [None, None, None, None] # label, score, bbox, mask
        
        self.idx = 0
        self.depth_unit = 1000

        self._lock = threading.RLock()

        msg = rospy.wait_for_message(args.caminfo_channel, CameraInfo)
        self.K = np.array(msg.K).reshape(3, 3)
        self.K_updated = False

        self.last_save_pose = None
        
        rgb_sub = Subscriber(args.rgb_channel, Image)
        depth_sub = Subscriber(args.depth_channel, Image)
        ts = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.5)
        ts.registerCallback(self.img_callback)

        # self.meta_sub = rospy.Subscriber(args.input_meta_channel, Float32MultiArray, self.maskrcnn_meta_callback)
        # self.mask_sub = rospy.Subscriber(args.input_mask_channel, Float32MultiArray, self.maskrcnn_mask_callback)
        meta_sub = Subscriber(args.input_meta_channel, Float32MultiArray)
        mask_sub = Subscriber(args.input_mask_channel, Float32MultiArray)
        ts1 = ApproximateTimeSynchronizer([meta_sub, mask_sub], 10, 0.5, allow_headerless=True)
        ts1.registerCallback(self.maskrcnn_meta_mask_callback)

        # self.calib_sub = rospy.Subscriber(args.cam2robot_calib_channel, PoseStamped, self.register_calib_callback, queue_size=1)

        self.save_sub = rospy.Subscriber(args.user_save_channel, Bool, self.savefile, queue_size=1)

        self.vis_pub = rospy.Publisher(args.output_vis_channel, Image, queue_size=1)
        self.pose_pub = rospy.Publisher(args.output_pose_channel, Float32MultiArray, queue_size=1)

        if args.alwaysrun_tag:
            rospy.loginfo(f'always run inference in {args.hz} Hz')
        else: # will only run when activated
            rospy.loginfo(f'Waiting activation from {args.user_activate_channel} to run inference')
            self.run_net_sub = rospy.Subscriber(args.user_activate_channel, Bool, self.inference, queue_size=1)
    
    def crop_img(self):
        with self._lock: # TODO: only work for 720*1280 input
            self._rgb_image = cv2.resize(self._rgb_image[:, 160:1120], (640, 480), interpolation=cv2.INTER_NEAREST)
            self._depth_image = cv2.resize(self._depth_image[:, 160:1120], (640, 480), interpolation=cv2.INTER_NEAREST)
            if not self.K_updated:
                self.K = np.array([
                    [self.K[0, 0] * 2 / 3, 0, (self.K[0, 2]-160) * 2 / 3],
                    [0, self.K[1, 1] * 2 / 3, self.K[1, 2] * 2 / 3],
                    [0., 0., 1.]
                ])
                self.K_updated = True
                rospy.loginfo('intrinsic updated to')
                print(self.K)
 
    def img_callback(self, rgb, depth):
        with self._lock:
            self._rgb_image = ros_numpy.numpify(rgb)
            self._depth_image = ros_numpy.numpify(depth).astype(np.float64) / self.depth_unit
            if self._rgb_image.shape[0] != 480:
                self.crop_img()
    
    def maskrcnn_meta_callback(self, metas):
        with self._lock:
            # metas: [N, 6] -> labels, scores, boxes
            metas = np.array(metas.data).reshape(-1, 6)
            labels, scores, boxes = metas[:, 0].astype(np.uint16), metas[:, 1], metas[:, 2:]
            self.maskrcnn_output[0] = labels
            self.maskrcnn_output[1] = scores
            self.maskrcnn_output[2] = boxes
    
    def maskrcnn_mask_callback(self, masks):
        with self._lock:
            masks = np.array(masks.data).reshape(-1, 480, 640) # TODO: add image resolution to config
            self.maskrcnn_output[3] = masks

    def maskrcnn_meta_mask_callback(self, metas, masks):
        with self._lock:
            # metas: [N, 6] -> labels, scores, boxes
            metas = np.array(metas.data).reshape(-1, 6)
            labels, scores, boxes = metas[:, 0].astype(np.uint16), metas[:, 1], metas[:, 2:]
            self.maskrcnn_output[0] = labels
            self.maskrcnn_output[1] = scores
            self.maskrcnn_output[2] = boxes
            
            masks = np.array(masks.data).reshape(-1, 480, 640) # TODO: add image resolution to config
            self.maskrcnn_output[3] = masks
    
    def register_calib_callback(self, msg):
        # receive transformation from camera to robot's chest (added transform to have z upward, x forward in robot world), then fit plane using depth image
        if self.Tcam2robot is not None:
            return
        with self._lock:
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            rotmat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.identity(4)
            T[:3, :3] = rotmat
            T[:3, 3] = [x, y, z]
            self.Tcam2robot = np.linalg.inv(T)
            rospy.loginfo('Received cam2robot calibration, will apply to object pose estimates')

    def inference(self, msg): # for user activation
        if self._rgb_image is None or self._depth_image is None:
            rospy.loginfo('Not receive RGB-D image')
            return
        if self.maskrcnn_output[0] is None:
            rospy.loginfo('Not receive mask-rcnn output')
            return
        with self._lock:
            output = {}
            labels, scores, boxes, masks = self.maskrcnn_output
            # print(labels.shape, scores.shape, boxes.shape, masks.shape)
            if len(masks) != len(boxes):
                rospy.logwarn('mask and label-box-score not from the same image, skip inference') # TODO: add control
                print(labels.shape, scores.shape, boxes.shape, masks.shape)
                return

            output['scale'] = []
            output['Trans'] = []
            output['labels'] = []
            output['masks'] = []
            output['boxes'] = []

            coord_2d = get_2d_coord_np(self.dataset.resolution[0], self.dataset.resolution[1]).transpose(1, 2, 0)
            coord_2d_homo = np.concatenate((coord_2d, np.ones_like(coord_2d[..., [0]])), axis = 2).transpose((2, 0, 1))
            raydirection = np.einsum('ij, jkl->ikl', np.linalg.inv(self.K), coord_2d_homo).transpose((1, 2, 0))
            raydirection = raydirection/np.linalg.norm(raydirection, axis = 2, keepdims = True)
            transparent_mask = np.ones_like(self._depth_image)

            for idx in range(len(boxes)):
                bbx = boxes[idx]
                label = labels[idx]
                score = scores[idx]
                mask = masks[idx].squeeze()        
                if score < self.args.score_threshold:
                    continue
                if mask.sum() < self.args.pixelthreshold:
                    continue        
                transparent_mask[masks[0].squeeze() == 1] = 0
            
            rgbinputs = []
            pcdinputs = []
            Ks = []
            fxs = []
            fys = []
            choose_uvs = []
            cat_ids = []
            rotations = []
            translations = []
            fsnet_scales = []
            sym_infos = []
            mean_shapes = []
            dpt_gt_chooses = []  
            dpt_pcds = []
            dpt_gt_pcds = []
            raydirection_patchs = []
            dpt_gt_patchs = []
            roi_mask_patchs = []
            roi_transparent_masks = []
            gt_bbxs = []
            # shape_priors = []
            for idx in range(len(boxes)):
                bbx = boxes[idx]
                bbx = torch.tensor(bbx)
                label = labels[idx]
                score = scores[idx]
                mask = masks[idx].squeeze()
                mask = torch.tensor(mask)    
                if score < self.args.score_threshold:
                    continue
                if mask.sum() < self.args.pixelthreshold:
                    continue
                cat_id = label.item() - 1
                sym_info = self.dataset.get_sym_info_bycate(cat_id)
                mean_shape = self.dataset.cate_meanscale[label]["scale"]
                bbx = bbx.cpu().detach().numpy()
                roi_mask = mask.cpu().detach().unsqueeze(dim=2).numpy()
                rgb = self._rgb_image.astype(np.float32) / 255
                rgbinput, bbox_center, scale = self.dataset.rgbinput(rgb, self._depth_image, self._depth_image, rgb, coord_2d, bbx)
                raydirection_patch = self.dataset.get_patch(raydirection, bbox_center, scale)
                mask = np.asarray(mask.unsqueeze(dim=2))
                roi_transparent_mask = self.dataset.get_patch(mask, bbox_center, scale)
                index = np.where(raydirection_patch[:, :, [2]] == 0)
                raydirection_patch[index[0], index[1], np.ones_like(index[2])*2] = 1
                raydirection_patch = raydirection_patch/raydirection_patch[:, :, [2]]
                fx =  self.K[0, 0]
                fy =  self.K[1, 1]
                new_bbx_mask = (coord_2d[..., 0] - bbox_center[0] <= scale/2 - 1) *\
                            (coord_2d[..., 0] - bbox_center[0] >= -scale/2) *\
                            (coord_2d[..., 1] - bbox_center[1] <= scale/2 - 1) *\
                            (coord_2d[..., 1] - bbox_center[1] >= -scale/2)
                dpt_gt_patch = self.dataset.get_patch(self._depth_image[:, :, np.newaxis], bbox_center, scale)
                
                roi_mask_patch = self.dataset.get_patch(roi_mask, bbox_center, scale)
                all_pointu, all_pointv= np.where(roi_mask.astype(bool).squeeze() * (self._depth_image > 0) * new_bbx_mask)
                if len(all_pointu) == 0:
                    continue
                ## when number of mask pixel less then self.args.pcdnum_input, use replace to choose
                choose_index = np.random.choice(len(all_pointu), self.args.pcdnum_input, replace=len(all_pointu) <= self.args.pcdnum_input)
                choose_point_uv = (all_pointu[choose_index], all_pointv[choose_index])
                ## choose_point_uv_revelant is UV revelant to the bbxs, minus the top, left pixel of the generate bbx
                choose_point_uv_revelant = ((all_pointu[choose_index] - bbox_center[1] + scale/2)*(self.args.rgbinputdim[0]-1)/(scale-1), #480 
                                            (all_pointv[choose_index] - bbox_center[0] + scale/2)*(self.args.rgbinputdim[1]-1)/(scale-1)) #640
                if choose_point_uv_revelant[0].max() + 1 >= self.args.rgbinputdim[0] - 1 or choose_point_uv_revelant[0].min() < 0:
                    continue
                if choose_point_uv_revelant[1].max() + 1 >= self.args.rgbinputdim[1] - 1 or choose_point_uv_revelant[1].min() < 0:
                    continue             
                choose_uv = np.stack(choose_point_uv_revelant, axis = 1)
                pcdinput = self.dataset.pcdinput(rgb, self._depth_image, self._depth_image, self._depth_image, rgb, coord_2d, raydirection, choose_point_uv) # [pcdinputdim, c_pcd]
                dpt_gt_choose = self._depth_image[choose_point_uv[0], choose_point_uv[1]]
                dpt_pcd = (self._depth_image[:, :, np.newaxis] * raydirection/raydirection[:, :, [2]])[choose_point_uv[0], choose_point_uv[1], :]
                dpt_gt_pcd = (self._depth_image[:, :, np.newaxis] * raydirection/raydirection[:, :, [2]])[choose_point_uv[0], choose_point_uv[1], :]
                ## rescale UV map to [0, 1]
                # coord_2d[..., 0] = coord_2d[..., 0]/self.resolution[0]
                # coord_2d[..., 1] = coord_2d[..., 1]/self.resolution[1]
                rmin, rmax, cmin, cmax = get_bbox(bbx)
                bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
                output['labels'].append(label)
                output['masks'].append(mask)
                output['boxes'].append(bbx)
                rgbinputs.append(rgbinput)
                pcdinputs.append(pcdinput)
                # Ks.append(K)
                fxs.append(np.array([fx]))
                fys.append(np.array([fy]))
                choose_uvs.append(choose_uv)
                cat_ids.append(cat_id)
                # obj_ids_0base.append(cat_id + 1)
                mean_shapes.append(mean_shape)
                dpt_gt_chooses.append(dpt_gt_choose) 
                # obj_poses.append(obj_pose)
                # obj_scales.append(obj_scale)
                sym_infos.append(sym_info)
                gt_bbxs.append(bbox_xyxy)
                dpt_pcds.append(dpt_pcd)
                dpt_gt_pcds.append(dpt_gt_pcd)
                raydirection_patchs.append(raydirection_patch)
                dpt_gt_patchs.append(dpt_gt_patch)
                roi_mask_patchs.append(roi_mask_patch)
                roi_transparent_masks.append(roi_transparent_mask)        
                # shape_priors.append(np.array(self.dataset.shapeprior[cat_id]))

            rgbinputs = np.array(rgbinputs)
            pcdinputs = np.array(pcdinputs)
            Ks = np.array(Ks)
            fxs = np.array(fxs)
            fys = np.array(fys)
            choose_uvs = np.array(choose_uvs)
            cat_ids = np.array(cat_ids)
            rotations = np.array(rotations)
            translations = np.array(translations)
            fsnet_scales = np.array(fsnet_scales)
            sym_infos = np.array(sym_infos)
            mean_shapes = np.array(mean_shapes)
            dpt_gt_chooses = np.array(dpt_gt_chooses)
            gt_bbxs = np.array(gt_bbxs)
            dpt_pcds = np.array(dpt_pcds)
            dpt_gt_pcds = np.array(dpt_gt_pcds)
            raydirection_patchs = np.array(raydirection_patchs)
            dpt_gt_patchs = np.array(dpt_gt_patchs)
            roi_mask_patchs = np.array(roi_mask_patchs)
            roi_transparent_masks = np.array(roi_transparent_masks)
            # shape_priors = np.array(shape_priors)

            data_dict = {}

            data_dict['rgbinput'] = torch.as_tensor(rgbinputs.astype(np.float32)).contiguous()
            data_dict['pcdinput'] = torch.as_tensor(pcdinputs.astype(np.float32)).contiguous()
            # data_dict['cam_K'] = torch.as_tensor(Ks.astype(np.float32)).contiguous()
            data_dict['fx'] = torch.as_tensor(fxs.astype(np.float32)).contiguous()
            data_dict['fy'] = torch.as_tensor(fys.astype(np.float32)).contiguous()
            data_dict['choose_uv'] = torch.as_tensor(choose_uvs.astype(np.float32)).contiguous()
            data_dict['cat_id'] = torch.as_tensor(cat_ids, dtype=torch.int64).contiguous()
            data_dict['sym_info'] = torch.as_tensor(sym_infos.astype(np.float32)).contiguous()
            data_dict['mean_shape'] = torch.as_tensor(mean_shapes, dtype=torch.float32).contiguous()
            data_dict['dpt_pcd'] = torch.as_tensor(dpt_pcds, dtype=torch.float32).contiguous()
            data_dict['dpt_gt'] = torch.as_tensor(dpt_gt_chooses, dtype=torch.float32).contiguous()
            data_dict['dpt_gt_pcd'] = torch.as_tensor(dpt_gt_pcds, dtype=torch.float32).contiguous()
            data_dict['raydirection_patch'] = torch.as_tensor(raydirection_patchs, dtype=torch.float32).contiguous()
            data_dict['dpt_gt_patch'] = torch.as_tensor(dpt_gt_patchs, dtype=torch.float32).contiguous()
            data_dict['roi_mask_patch'] = torch.as_tensor(roi_mask_patchs, dtype=torch.bool).contiguous() 
            data_dict['roi_transparent_masks'] = torch.as_tensor(roi_transparent_masks, dtype=torch.bool).contiguous()
            # data_dict['shape_prior'] = torch.as_tensor(shape_priors, dtype=torch.float32).contiguous()

            for obj_id in range(len(data_dict['cat_id'])):
                data_dict_perobj = {}
                for item in data_dict:
                    data_dict_perobj[item] = data_dict[item][[obj_id]].cuda()
                output_dict_perobj = self.transnet_model(data_dict_perobj)
                output['Trans'].append(output_dict_perobj['Pred_RT'].cpu().numpy().squeeze())
                output['scale'].append(output_dict_perobj['Pred_s'].cpu().numpy().squeeze())
            output['scale'] = np.array(output['scale']) # (N, 3)
            output['Trans'] = np.array(output['Trans']) # (N, 4, 4)
            output['labels'] = np.array(output['labels']) # (N, 1)
            output['masks'] = np.array(output['masks']) # (N, 480, 640)
            output['boxes'] = np.array(output['boxes']) # (N, 4)
            self.output = output
            
            self.idx += 1
            print(self.idx)

    def publish(self):
        with self._lock:
            if self.vis_img is not None:
                self.vis_pub.publish(ros_numpy.msgify(Image, self.vis_img, encoding='bgr8'))
            if os.path.exists(self.args.savefile_path + f'transnet_output.pt'):
                res = torch.load(self.args.savefile_path + f'transnet_output.pt')
                if self.Tcam2robot is not None: # transform to robot frame after calibration
                    for i in range(len(res['Trans'])):
                        res['Trans'][i] = res['Trans'][i] @ self.Tcam2robot
                
                pose_msg = Float32MultiArray(data=res['Trans'].flatten()) # [N, 4, 4]
                self.pose_pub.publish(pose_msg)
            if self.args.visualize_tag:
                self.visualize()
            # if self.args.savefile_tag:
            #     self.savefile(None)

    def savefile(self, msg): # for user activation
        if self.output is None:
            return
        torch.save(self.output, self.args.savefile_path + f'transnet_output.pt')
        if self.vis_img is not None:
            cv2.imwrite(self.args.savefile_path + f'transnet_3Dbbox.png', self.vis_img)
        rospy.loginfo(f'result saved to transnet_output.pt')
    
    def visualize(self):
        if self.output is None:
            return
        scale = self.output['scale']
        pose = self.output['Trans']
        print_msg = f'{len(scale)} obj pose estimates'
        if self.Tcam2robot is not None:
            print_msg += ', transformed to robot frame using calibration'
        print(print_msg)
        self.vis_img = deepcopy(self._rgb_image[:, :, ::-1])
        for i in range(len(scale)):
            self.vis_3dbbox(i, scale[i], pose[i])
            self.vis_grasp_pose(pose[i])
    
    def vis_3dbbox(self, idx, scale, pose):
        x, y, z = scale[0] / 2, scale[1] / 2, scale[2] / 2
        p = []
        p.append(np.matmul(pose, np.array([x, y, z, 1]))[:3])
        p.append(np.matmul(pose, np.array([-x, y, z, 1]))[:3])
        p.append(np.matmul(pose, np.array([-x, -y, z, 1]))[:3])
        p.append(np.matmul(pose, np.array([x, -y, z, 1]))[:3])
        p.append(np.matmul(pose, np.array([x, y, -z, 1]))[:3])
        p.append(np.matmul(pose, np.array([-x, y, -z, 1]))[:3])
        p.append(np.matmul(pose, np.array([-x, -y, -z, 1]))[:3])
        p.append(np.matmul(pose, np.array([x, -y, -z, 1]))[:3])
        p = np.asarray(p)
        points = (self.K @ p.T).T
        points = points / points[:, [2]]
        img_p = []
        for point in points:
            img_p.append(tuple((int(point[0]), int(point[1]))))
        self.vis_img = cv2.line(self.vis_img, img_p[0], img_p[1], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[1], img_p[2], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[2], img_p[3], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[3], img_p[0], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[4], img_p[5], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[5], img_p[6], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[6], img_p[7], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[7], img_p[4], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[0], img_p[4], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[1], img_p[5], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[2], img_p[6], (255, 0, 0))
        self.vis_img = cv2.line(self.vis_img, img_p[3], img_p[7], (255, 0, 0))
        text_xy = [0, 0]
        text_xy[0] = int(img_p[0][0])
        text_xy[1] = int(img_p[0][1])
        self.vis_img = cv2.putText(self.vis_img, str(idx), (text_xy[0], text_xy[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(0,255,0))

    def vis_grasp_pose(self, pose, offset=0.05):
        # print('in vis')
        x_z, y_z = pose[0, 3] / pose[2, 3], pose[1, 3] / pose[2, 3]
        uv1 = (np.array(self.K) @ np.array([x_z, y_z, 1])).astype('int')
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for j in range(3): # add to center along 3 axes
            tmp = pose[:3, 3] + offset * pose[:3, j]
            x_z, y_z = tmp[0] / tmp[2], tmp[1] / tmp[2]
            uv2 = (np.array(self.K) @ np.array([x_z, y_z, 1])).astype('int')
            self.vis_img = cv2.line(self.vis_img, (uv1[0], uv1[1]), (uv2[0], uv2[1]), colors[j], 3)
        self.vis_img = cv2.circle(self.vis_img, (uv1[0], uv1[1]), 5, (255, 255, 255))


def main(argv):
    
    args = flags.FLAGS
    node = TransNet_Pose_Estimator(args)

    rospy.loginfo(f'{args.node_name} node started')
    r = rospy.Rate(args.hz)

    while not rospy.is_shutdown():
        if args.alwaysrun_tag:
            node.inference(None)
        r.sleep()
        node.publish()

if __name__ == "__main__":
    # cannot use argparse together with absl.flags
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--node_name', type=str, default='TransNet_Pose_Estimator')
    # parser.add_argument('--visualize_tag', type=bool, default=True, help='when set to true, will draw 3D bbox and pose for estimated object')
    # parser.add_argument('--savefile_tag', type=bool, default=True, help='when set to true, will save all image and detected metadata to savefile_path')
    # parser.add_argument('--savefile_path', type=str, default='data/output/')
    # parser.add_argument('--alwaysrun_tag', type=bool, default=False, help='when set to true, will run in constant frequency --hz, otherwise will run once received activation signal from --user_activate_channel')
    # parser.add_argument('--hz', type=float, default=1)
    # parser.add_argument('--category', type=str, default='water_cup')
    # parser.add_argument('--transnet_weight_path', type=str, default='src/transnet_ros/src/weight/water_cup_model_26.pt', help='reminder to check consistency with --category')

    # parser.add_argument('--rgb_channel', type=str, default='/camera/color/image_raw')
    # parser.add_argument('--depth_channel', type=str, default='/camera/aligned_depth_to_color/image_raw')
    # parser.add_argument('--caminfo_channel', type=str, default='/camera/color/camera_info')
    # parser.add_argument('--input_meta_channel', type=str, default='/maskrcnn_out_meta', help='check consistency with mask_rcnn_detector node')
    # parser.add_argument('--input_mask_channel', type=str, default='/maskrcnn_out_mask')
    # parser.add_argument('--cam2robot_calib_channel', type=str, default='/cam2robot_transform', help='check consistency with camera_calibrator node')
    # parser.add_argument('--output_vis_channel', type=str, default='/transnet_vis')
    # parser.add_argument('--output_pose_channel', type=str, default='/transnet_out_pose')
    # parser.add_argument('--user_activate_channel', type=str, default='/transnet_activate')

    # args = parser.parse_args()

    flags.DEFINE_string('node_name', 'TransNet_Pose_Estimator', '')
    flags.DEFINE_bool('visualize_tag', True, 'when set to true, will draw 3D bbox and pose for estimated object')
    flags.DEFINE_bool('savefile_tag', True, 'when set to true, will save all image and detected metadata to savefile_path')
    flags.DEFINE_string('savefile_path', 'data/output/', '')
    flags.DEFINE_bool('alwaysrun_tag', True, 'when set to true, will run in constant frequency --hz, otherwise will run once received activation signal from --user_activate_channel')
    flags.DEFINE_float('hz', 0.5, '')
    flags.DEFINE_float('score_threshold', 0.5, '')
    flags.DEFINE_string('category', 'water_cup', '')
    # flags.DEFINE_string('transnet_weight_path', 'src/transnet_ros/src/weight/water_cup_model_26.pth', 'check consistency with --category')
    flags.DEFINE_string('transnet_weight_path', 'src/transnet_ros/src/model/TransNet/wine_cup_new_model_29.pth', 'check consistency with --category')
    flags.DEFINE_string('rgb_channel', '/camera/color/image_raw', '')
    flags.DEFINE_string('depth_channel', '/camera/aligned_depth_to_color/image_raw', '')
    flags.DEFINE_string('caminfo_channel', '/camera/color/camera_info', '')
    flags.DEFINE_string('input_meta_channel', '/maskrcnn_out_meta', 'check consistency with mask_rcnn_detector node')
    flags.DEFINE_string('input_mask_channel', '/maskrcnn_out_mask', 'check consistency with mask_rcnn_detector node')
    flags.DEFINE_string('cam2robot_calib_channel', '/cam2robot_transform', 'check consistency with camera_calibrator node')
    flags.DEFINE_string('output_vis_channel', '/transnet_vis', '')
    flags.DEFINE_string('output_pose_channel', '/transnet_out_pose', '')
    flags.DEFINE_string('user_activate_channel', '/transnet_activate', '')
    flags.DEFINE_string('user_save_channel', '/transnet_save', 'will update last saved estimation and keep publish it')

    app.run(main)