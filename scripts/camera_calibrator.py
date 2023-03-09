from dt_apriltags import Detector
import cv2
import numpy as np
import open3d as o3d
import g2o
from copy import deepcopy
import argparse
import threading
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
import ros_numpy


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().set_verbose(True)
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


def gettransform(tagcam, tagrobot):
    ## return the pose of camera in robot camera coordinate
    T_tags_cam = np.hstack((tagcam.pose_R, tagcam.pose_t))
    T_tags_cam = np.vstack((T_tags_cam, np.array([[0, 0, 0, 1]])))
    T_tags_robot = np.hstack((tagrobot.pose_R, tagrobot.pose_t))
    T_tags_robot = np.vstack((T_tags_robot, np.array([[0, 0, 0, 1]])))
    T_cam_robot = T_tags_robot.dot(np.linalg.inv(T_tags_cam))
    return T_cam_robot, T_tags_cam, T_tags_robot

def pose44_g2o(pose44):
    return g2o.Isometry3d(pose44[:3, :3], pose44[:3, 3])

def g2o_pose44(g2o_pose):
    rot_mat = g2o_pose.orientation().R
    pos = g2o_pose.position()
    pose44 = np.identity(4)
    pose44[:3, :3] = rot_mat
    pose44[:3, 3] = pos
    return pose44

def vis_camposes(poses0, poses1):
    mesh_frame0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
    frames0, frames1 = [], []
    for i in range(len(poses0)):
        frames0.append(deepcopy(mesh_frame0).transform(poses0[i]))
        frames1.append(deepcopy(mesh_frame1).transform(poses1[i]))
    o3d.visualization.draw_geometries(frames0 + frames1)


class CameraCalibrator():
    def __init__(self, args):

        rospy.init_node(args.node_name)

        self.args = args

        self.dt = Detector(searchpath=['apriltags'],
                    # families='tagStandard41h12',
                    families=args.tag_family,
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

        self.opt = PoseGraphOptimization()

        self._lock = threading.RLock()

        rospy.loginfo(f'Waiting for camera info from {args.cam1_info_channel} and {args.cam2_info_channel}')
        msg = rospy.wait_for_message(args.cam1_info_channel, CameraInfo)
        self.cam_K1 = np.array(msg.K).reshape(3, 3)
        msg = rospy.wait_for_message(args.cam2_info_channel, CameraInfo)
        self.cam_K2 = np.array(msg.K).reshape(3, 3)


        self.rgb_sub1 = rospy.Subscriber(args.cam1_rgb_channel, Image, self.img_callback1)
        self.rgb_sub2 = rospy.Subscriber(args.cam2_rgb_channel, Image, self.img_callback2)

        self.run_calib_sub = rospy.Subscriber(args.user_activate_channel, Bool, self.calibrate, queue_size=1)
        self.trans_pub = rospy.Publisher(args.output_channel, PoseStamped, queue_size=1)

        self._rgb_image1 = None
        self._rgb_image2 = None
        self.tags1 = None
        self.tags2 = None
        self.match_tagids = None
        self.vis_img1 = None
        self.vis_img2 = None
        self.vis_tag1to2 = None
        self.cam1_to_cam2 = None

        if not args.run_once:
            rospy.loginfo(f'Waiting activation from {args.user_activate_channel} to run calibration')


    def img_callback1(self, img):
        with self._lock:
            self._rgb_image1 = ros_numpy.numpify(img)
            # rospy.loginfo(f'received image from {args.cam1_rgb_channel}')

    def img_callback2(self, img):
        with self._lock:
            self._rgb_image2 = ros_numpy.numpify(img)
            # rospy.loginfo(f'received image from {args.cam2_rgb_channel}')

    def calibrate(self, msg): # msg is for user activation
        with self._lock:
            if self._rgb_image1 is None or self._rgb_image2 is None:
                rospy.loginfo('No RGB image pair received')
                return False
            self.tags1 = self.dt.detect(cv2.cvtColor(self._rgb_image1, cv2.COLOR_RGB2GRAY), True, (self.cam_K1[0, 0], self.cam_K1[1, 1], self.cam_K1[0, 2], self.cam_K1[1, 2]), self.args.tag_size)
            self.tags2 = self.dt.detect(cv2.cvtColor(self._rgb_image2, cv2.COLOR_RGB2GRAY), True, (self.cam_K2[0, 0], self.cam_K2[1, 1], self.cam_K2[0, 2], self.cam_K2[1, 2]), self.args.tag_size)
            self.match_tagids = ([], [])
            for i1, t1 in enumerate(self.tags1):
                for i2, t2 in enumerate(self.tags2):
                    if t1.tag_id == t2.tag_id:
                        self.match_tagids[0].append(i1)
                        self.match_tagids[1].append(i2)
            
            self.opt.add_vertex(0, pose44_g2o(np.identity(4))) # node for cam1
            for i in range(len(self.match_tagids[0])):
                tag1, tag2 = self.tags1[self.match_tagids[0][i]], self.tags2[self.match_tagids[1][i]]
                T_cam1tocam2, T_cam1, T_cam2 = gettransform(tag1, tag2)
                if i == 0:
                    self.opt.add_vertex(1, pose44_g2o(np.linalg.inv(T_cam1tocam2))) # node for cam2
                self.opt.add_vertex(i + 2, pose44_g2o(T_cam1)) # node for tag
                self.opt.add_edge([0, i + 2], pose44_g2o(T_cam1)) # edge for cam1-tag
                self.opt.add_edge([1, i + 2], pose44_g2o(T_cam2)) # edge for cam2-tag
            self.opt.optimize()

            cam1_pose = g2o_pose44(self.opt.get_pose(0))
            cam2_pose = g2o_pose44(self.opt.get_pose(1))
            self.cam1_to_cam2 = cam1_pose.dot(np.linalg.inv(cam2_pose))

            rospy.loginfo(f'Calibration done, cam1 to cam2 transformation matrix:')
            print(self.cam1_to_cam2)

            if self.args.visualize_tag:
                self.visualize()

            if self.args.savefile_tag:
                self.savefile()

            return True

    def publish(self):
        if self.cam1_to_cam2 is not None:
            p = PoseStamped()
            p.header.seq = 0
            p.header.stamp = rospy.Time.now()
            p.header.frame_id = 'base'
            p.pose.position.x = self.cam1_to_cam2[0, 3]
            p.pose.position.y = self.cam1_to_cam2[1, 3]
            p.pose.position.z = self.cam1_to_cam2[2, 3]
            quat = R.from_matrix(self.cam1_to_cam2[:3, :3]).as_quat()
            p.pose.orientation.x = quat[0]
            p.pose.orientation.y = quat[1]
            p.pose.orientation.z = quat[2]
            p.pose.orientation.w = quat[3]
            self.trans_pub.publish(p)

    def visualize(self):
        if self.tags1 is not None:
            self.vis_img1 = deepcopy(self._rgb_image1[:, :, ::-1])
            for tag in self.tags1:
                cv2.circle(self.vis_img1, tuple((int(tag.center[0]), int(tag.center[1]))), 2, (0, 255, 0), 2)
                for corner in tag.corners:
                    cv2.circle(self.vis_img1, tuple((int(corner[0]), int(corner[1]))), 2, (0, 255, 0), 2)
        if self.tags2 is not None:
            self.vis_img2 = deepcopy(self._rgb_image2[:, :, ::-1])
            for tag in self.tags2:
                cv2.circle(self.vis_img2, tuple((int(tag.center[0]), int(tag.center[1]))), 2, (0, 255, 0), 2)
                for corner in tag.corners:
                    cv2.circle(self.vis_img2, tuple((int(corner[0]), int(corner[1]))), 2, (0, 255, 0), 2)
        if self.cam1_to_cam2 is not None:
            tag_pose_list = []
            for idx in range(len(self.match_tagids[0])):
                tag_pose_list.append(g2o_pose44(self.opt.get_pose(idx + 2)))

            points_cam1_list = []
            tag_size = self.args.tag_size
            points_intag = np.array([
                [0, 0, 0],
                [tag_size/2, tag_size/2, 0],
                [-tag_size/2, tag_size/2, 0],
                [tag_size/2, -tag_size/2, 0],
                [-tag_size/2, -tag_size/2, 0],
            ]).T
            for tag_pose in tag_pose_list:
                points_cam1_list.append(tag_pose[:3, :3] @ points_intag + tag_pose[:3, [3]])

            points_cam2 = self.cam1_to_cam2[:3, :3] @ np.hstack(points_cam1_list) + self.cam1_to_cam2[:3, [3]]
            pixel_homo = self.cam_K2 @ points_cam2
            pixel = pixel_homo[:2] / pixel_homo[[2]]

            self.vis_tag1to2 = deepcopy(self._rgb_image2)
            for idx in range(pixel.shape[1]):
                cv2.circle(self.vis_tag1to2, tuple((int(pixel[0, idx]), int(pixel[1, idx]))), 2, (0, 255, 0), 2)
    
    def savefile(self):
        if self.vis_img1 is not None:
            cv2.imwrite(self.args.savefile_path + 'tag_vis1.png', self.vis_img1)
        if self.vis_img2 is not None:
            cv2.imwrite(self.args.savefile_path + 'tag_vis2.png', self.vis_img2)
        if self.vis_tag1to2 is not None:
            cv2.imwrite(self.args.savefile_path + 'tag_vis_proj1to2.png', self.vis_img2)
        if self.cam1_to_cam2 is not None:
            with open(self.args.savefile_path + 'trans.npy', 'wb') as f:
                np.save(f, np.array(self.cam1_to_cam2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_name', type=str, default='Camera_Calibrator')

    parser.add_argument('--tag_size', type=float, default=0.1385)
    parser.add_argument('--tag_family', type=str, default='tag36h11')

    parser.add_argument('--cam1_rgb_channel', type=str, default='/camera/color/image_raw')
    parser.add_argument('--cam1_info_channel', type=str, default='/camera/color/camera_info')
    parser.add_argument('--cam2_rgb_channel', type=str, default='/head_camera/rgb/image_raw')
    parser.add_argument('--cam2_info_channel', type=str, default='/head_camera/rgb/camera_info')
    parser.add_argument('--user_activate_channel', type=str, default='/calib_activate')
    parser.add_argument('--output_channel', type=str, default='/cam2robot_transform')
    parser.add_argument('--visualize_tag', type=bool, default=True, help='when set to true, will draw detected tag points on images')
    parser.add_argument('--savefile_tag', type=bool, default=True, help='when set to true, will save transform and all images to savefile_path, see savefile() function for detailed name suffix')
    parser.add_argument('--savefile_path', type=str, default='data/output/')
    parser.add_argument('--run_once', type=bool, default=True, help='when set to true, will automatically run calibrate once and keep publishing the transform, won\'t wait for user activation')

    
    args = parser.parse_args()

    p = CameraCalibrator(args)
    calibrated = False
    while not rospy.is_shutdown():
        rospy.sleep(0.01)
        if args.run_once and not calibrated:
            res = p.calibrate(None)
            if res:
                calibrated = True
        p.publish()
