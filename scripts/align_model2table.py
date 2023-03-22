import torch


def align_model2table(depth, intrinsic, pose_dict, bbx_dict):
    """_summary_
    Args:
        depth (_type_): W*H
        intrinsic (_type_): 3*3 
        pose_dict (_type_): N*3*4
        bbx_dict : N * 3
    """
    
    align_pose_dict = torch.zeros_like(pose_dict)
    plane_model, _, _ = fit_plane(depth, intrinsic)
    print(plane_model)
    v_direction = torch.tensor(plane_model[:3], dtype = torch.float32)
    for i in range(pose_dict.shape[0]):
        pose = pose_dict[i]
        z_direction = pose[:3, 2]
        x_direction = pose[:3, 0]
        y_direction = pose[:3, 1]
        location = pose[:3, 3]
        status = v_direction.dot(z_direction)
        new_pose = torch.zeros_like(pose)
        distance = location.dot(v_direction) + plane_model[3]
        if status >= 0.5:
            # object is vertical to the table
            ## set z to be 
            new_pose[:3, 2] = v_direction
            new_pose[:3, 0] = x_direction - v_direction.dot(x_direction) * v_direction
            new_pose[:3, 0] = new_pose[:3, 0]/torch.linalg.norm(new_pose[:3, 0])
            new_pose[:3, 1] = torch.cross(new_pose[:3, 2], new_pose[:3, 0])
            new_pose[:3, 3] = pose[:3, 3] + (bbx_dict[2]/2 - distance) * v_direction
            align_pose_dict[i] = new_pose
        elif status <= -0.5:
            # object is inverse-vertical to the table
            ## set z to be 
            new_pose[:3, 2] = -v_direction
            new_pose[:3, 0] = x_direction - v_direction.dot(x_direction) * v_direction
            new_pose[:3, 0] = new_pose[:3, 0]/torch.linalg.norm(new_pose[:3, 0])
            new_pose[:3, 1] = torch.cross(new_pose[:3, 2], new_pose[:3, 0])
            new_pose[:3, 3] = pose[:3, 3] + (bbx_dict[2]/2 - distance) * v_direction
            align_pose_dict[i] = new_pose
        else:
            # object is lying to the table
            new_pose[:3, 2] = z_direction - v_direction.dot(z_direction) * v_direction
            new_pose[:3, 2] = new_pose[:3, 2]/torch.linalg.norm(new_pose[:3, 2])
            new_pose[:3, 0] = v_direction
            new_pose[:3, 1] = torch.cross(new_pose[:3, 2], new_pose[:3, 0])
            new_pose[:3, 3] = pose[:3, 3] + (bbx_dict[0]/2 - distance) * v_direction
            align_pose_dict[i] = new_pose
        print(new_pose)
        print(v_direction)
    return align_pose_dict

def fit_plane(depth, Kmat):
    h, w = depth.shape
    u, v = np.meshgrid(np.array(range(w)), np.array(range(h)))
    xyz = (np.einsum('ij,jlk->ilk', np.linalg.inv(Kmat), np.stack((u, v, np.ones_like(u)))) * depth).reshape(3, -1).T
    xyz[xyz[:, 2] > 1] = 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    plane_pcd = downpcd.select_by_index(inliers)
    mesh_frame_plane = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    if plane_model[2] > 0:
        plane_model = -plane_model
    # visualize to verify
    plane_pcd.paint_uniform_color([0, 0, 0])
    pcd.paint_uniform_color([0, 1, 0])
    frame_pose = np.identity(4)
    frame_pose[:3, 3] = np.array(plane_pcd.points).mean(axis=0)
    frame_pose[:3, 2] = plane_model[:3]
    x = np.cross(np.array([0, 0, 1]), frame_pose[:3, 2])
    x = x / np.linalg.norm(x)
    frame_pose[:3, 0] = x
    frame_pose[:3, 1] = np.cross(frame_pose[:3, 2], x)
    mesh_frame_plane.transform(frame_pose)
    mesh_frame_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([downpcd, plane_pcd, mesh_frame_plane, mesh_frame_cam, pcd])
    return plane_model, plane_pcd, [downpcd, plane_pcd, mesh_frame_plane, mesh_frame_cam]

if __name__ == "__main__":
    import open3d as o3d
    import trimesh
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from PIL import Image
    depth_path = "/media/huijie/1E10C4A130358D19/clearpose/sample_1_1000/set3/scene4/000000-depth.png"
    with Image.open(depth_path) as li:
        depth = np.array(li)/1000
    intrinsic = torch.tensor([[601.3, 0.    , 334.7],
                            [0.   , 601.3 , 248.0],
                            [0.   , 0.    , 1.0]])
    pose_dict = torch.zeros(1, 3, 4)
    pose_dict[0, :3, :3] = torch.tensor(R.random().as_matrix())
    tm  = trimesh.load("/media/huijie/1E10C4A130358D19/clearpose/model/bowl_1/bowl_1.obj")
    plane_model, plane_pcd, origin_pcd = fit_plane(depth, intrinsic)
    plane_points = np.asarray(plane_pcd.points)
    squence = np.arange(plane_points.shape[0])
    np.random.shuffle(squence)
    plane_points = plane_points[squence, :]
    pose_dict[0, :3, 3] = torch.tensor(plane_points[squence, :][0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((np.array(pose_dict[0, :3, :3]) @ np.array(tm.vertices).T).T + np.array(pose_dict[0, :3, 3]))
    pcd.paint_uniform_color([1, 0, 0])
    

    bbx_dict = torch.tensor(np.array(tm.vertices).max(axis=0) - np.array(tm.vertices).min(axis=0))
    
    align_pose_dict = align_model2table(depth, intrinsic, pose_dict, bbx_dict)
    
    align_pcd = o3d.geometry.PointCloud()
    align_pcd.points = o3d.utility.Vector3dVector((np.array(align_pose_dict[0, :3, :3]) @ np.array(tm.vertices).T).T  + np.array(align_pose_dict[0, :3, 3]))
    align_pcd.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries(origin_pcd + [pcd, align_pcd])
    
    
    
    