import os

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
from typing import List, Optional

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from yacs.config import CfgNode

DEPTH_FILTER_MM = 1600  # None # TAKES PRIORITY OVER DEPTH_FILTER_PERCENTILE
DEPTH_FILTER_PERCENTILE = 95


def get_model_K(focal_length, img_size):
    img_w, img_h = img_size[:, 0], img_size[:, 1]

    # Create camera intrinsics matrix as a tensor
    K = torch.tensor([
        [focal_length, 0, img_w/2.],
        [0, focal_length, img_h/2.],
        [0, 0, 1]
    ], dtype=torch.float32)

    return K


def cam_crop_to_full(cam_bbox, box_center, box_size, K):
    # Convert cam_bbox to full image
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * K[0, 0] / bs
    tx = (2 * (cx - K[0, 2]) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - K[1, 2]) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def old_cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def project_full_img(points, cam_trans, K):
    points = points + cam_trans
    points = points / points[..., -1:]
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]


def full_image_to_3d(pixels, depths, K):
    pixels = torch.tensor(pixels, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    points = torch.zeros(pixels.shape[0], 3, dtype=torch.float32)
    depths = torch.tensor(depths.squeeze(), dtype=torch.float32)
    points[..., 0] = (pixels[..., 0] - K[0, 2]) * depths / K[0, 0]
    points[..., 1] = (pixels[..., 1] - K[1, 2]) * depths / K[1, 1]
    points[..., 2] = depths
    return points


class Renderer:

    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def __call__(self,
                 vertices: np.array,
                 camera_translation: np.array,
                 image: torch.Tensor,
                 full_frame: bool = False,
                 imgname: Optional[str] = None,
                 side_view=False, rot_angle=90,
                 mesh_base_color=(1.0, 1.0, 0.9),
                 scene_bg_color=(0, 0, 0),
                 return_rgba=False,
                 ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """

        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD,
                                                 device=image.device).reshape(3, 1, 1)
            image = image + \
                torch.tensor(self.cfg.MODEL.IMAGE_MEAN,
                             device=image.device).reshape(3, 1, 1)
            image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy(
            ) + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy(
            ) + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def vertices_to_trimesh_using_depth(self, vertices, camera_translation, depths, focal_length, img_res, mesh_base_color=(1.0, 1.0, 0.9), rot_axis=[1, 0, 0], rot_angle=0, is_right=1, K=None, hand_masks=None, K_model=None):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        # get wilor hand depths
        color, depths_render = self.render_rgba(vertices, camera_translation, rot_axis=rot_axis, rot=rot_angle,
                                                mesh_base_color=mesh_base_color, render_res=img_res, focal_length=K_model[0, 0], is_right=is_right, return_depth=True)

        # depth_render acts as a mask (identifies the 2D projected wilor hand) since points outside the mask have depth=0
        mask = (depths > 0) & (depths_render > 0)

        if not np.any(mask):
            print("No real depth for the rendered hand")
            return trimesh.Trimesh()

        # Apply additional hand mask if provided, selecting the closest hand mask to the projected wilor hand
        if hand_masks is not None:
            mask = mask.astype(bool)
            hand_masks = hand_masks.astype(bool)
            # Calculate centroid of reference mask
            ref_yx = np.array([coord.mean() for coord in np.nonzero(mask)])

            # Calculate centroids for all hand masks
            mask_centroids = np.array(
                [[coord.mean() for coord in np.nonzero(m[0])] for m in hand_masks])
            distances = np.linalg.norm(mask_centroids - ref_yx, axis=1)

            mask &= hand_masks[np.argmin(distances), 0]

        if not np.any(mask):
            print("No intersection between the rendered hand and the given hand mask")
            return trimesh.Trimesh()

        # Filter out points beyond a certain depth percentile.
        if DEPTH_FILTER_MM is None:
            # It's the percentile of the depth of the real points that match with the wilor hand
            depth_filter_percentile = np.percentile(
                depths[mask], DEPTH_FILTER_PERCENTILE)
            print(f"Depth filter percentile: {depth_filter_percentile}")
            mask &= (depths < depth_filter_percentile)
        else:
            mask &= (depths < DEPTH_FILTER_MM)

        if not np.any(mask):
            print("No intersection between the rendered hand and the depth filter")
            return trimesh.Trimesh()

        # Get the mean ratio of depth from visible vertices, which will be used to scale the wilor hand
        ratios = np.zeros_like(depths)
        ratios[mask] = depths[mask] / depths_render[mask]
        mean_depth_ratio = np.mean(ratios[mask])

        vertices_cam = vertices.copy() + camera_translation

        # Modify the depth of vertices_cam using the mean ratio of depth
        vertices_cam_2d = project_full_img(vertices_cam, [0, 0, 0], K_model)
        vertices_cam = full_image_to_3d(
            vertices_cam_2d, vertices_cam[..., -1:] * mean_depth_ratio, K)

        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        # transform depending on handedness
        if is_right:
            # process=False is necessary to avoid trimesh from removing duplicate vertices
            mesh = trimesh.Trimesh(vertices_cam, self.faces.copy(
            ), vertex_colors=vertex_colors, process=False)
        else:
            mesh = trimesh.Trimesh(vertices_cam, self.faces_left.copy(
            ), vertex_colors=vertex_colors, process=False)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
        self,
        vertices: np.array,
        cam_t=None,
        rot=None,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        camera_z=3,
        # camera_translation: np.array,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
        focal_length=None,
        is_right=None,
        return_depth=True,
        K=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        focal_length = focal_length if focal_length is not None else self.focal_length
        if K is not None:
            focal_length = K[0, 0]

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array(
                [0, 0, camera_z * focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array(
            [0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        if K is not None:
            camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1],
                                               cx=K[0, 2], cy=K[1, 2], zfar=1e12)
        else:
            camera_center = [render_res[0] / 2., render_res[1] / 2.]
            camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                               cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_depth:
            return color, rend_depth

        return color

    def render_rgba_multiple(
        self,
        vertices: List[np.array],
        cam_t: List[np.array],
        rot_axis=[1, 0, 0],
        rot_angle=0,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
        focal_length=None,
        is_right=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(
        ), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv, ttt, sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(
                    color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
