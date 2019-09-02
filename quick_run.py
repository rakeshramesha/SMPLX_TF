
import os, time
import os.path as osp
import argparse

import tensorflow as tf
import numpy as np

import pyrender
import trimesh
import smplx

def main(model_folder, model_type='smplx', ext='npz',
         gender='neutral', plot_joints=False,
         use_face_contour=False, out_fldr="./"):    

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         ext=ext)
    print(model)

    betas = tf.convert_to_tensor( np.random.standard_normal([1,10]).astype(np.float32))                 #[[-1.3573,  0.9423,  0.6675,  0.6804, -1.0229, -1.2455, -0.1667,  0.7943, 0.9591, 0.7925]]) 
    expression = tf.convert_to_tensor( np.random.standard_normal([1,10]).astype(np.float32))            #[[ 0.3169, -0.4887, -0.0194, -0.6570, -0.9348,  0.3143,  0.9349, -0.0479, 0.4494, -2.1038]]) 

    if (model_type.lower() == 'smpl'):
        ### 23 body pose joints in smpl while rest have 21 body pose joints
        bo_pose = tf.convert_to_tensor( np.random.standard_normal([1,23*3]).astype(np.float32)*0.1)     #tf.random_normal([1,23*3], dtype=tf.float32)*0.1
        output = model(betas=betas ,body_pose=bo_pose, return_verts=True)

    elif (model_type.lower() == 'smplh'):
        bo_pose = tf.convert_to_tensor( np.random.standard_normal([1,21*3]).astype(np.float32)*0.1)     #tf.random_normal([1,21*3], dtype=tf.float32)*0.1
        l_hand_pose = tf.convert_to_tensor( np.random.standard_normal([1,6]).astype(np.float32))        #[[-0.0869,  0.4552,  0.3587, -1.7560, -0.7198, -0.5980]]) #tf.random_normal([1, 6], dtype=tf.float32)
        r_hand_pose = tf.convert_to_tensor( np.random.standard_normal([1,6]).astype(np.float32))        #[[-0.0869,  0.4552,  0.3587, -1.7560, -0.7198, -0.5980]]) #tf.random_normal([1, 6], dtype=tf.float32)
        
        output = model(betas=betas, body_pose=bo_pose, left_hand_pose=l_hand_pose, right_hand_pose=r_hand_pose, return_verts=True)

    elif (model_type.lower() == 'smplx'):
        bo_pose = tf.convert_to_tensor( np.random.standard_normal([1,21*3]).astype(np.float32)*0.1)     #tf.random_normal([1,21*3], dtype=tf.float32)*0.1
        l_hand_pose = tf.convert_to_tensor( np.random.standard_normal([1,6]).astype(np.float32))        #[[-0.0869,  0.4552,  0.3587, -1.7560, -0.7198, -0.5980]]) #tf.random_normal([1, 6], dtype=tf.float32)
        r_hand_pose = tf.convert_to_tensor( np.random.standard_normal([1,6]).astype(np.float32))        #[[-0.0869,  0.4552,  0.3587, -1.7560, -0.7198, -0.59

        output = model(betas=betas, expression=expression, body_pose=bo_pose, left_hand_pose=l_hand_pose, right_hand_pose=r_hand_pose, return_verts=True)

    else:
        output = model(betas=betas, expression=expression,
                   return_verts=True)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess_1 = tf.InteractiveSession(config=config)
    sess_1.run(tf.global_variables_initializer())    

    vertices = output.vertices.eval(session=sess_1)
    joints = output.joints.eval(session=sess_1)

    vertices = np.squeeze(vertices)
    joints = np.squeeze(joints)

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
                               vertex_colors=vertex_colors)

    #rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    #tri_mesh.apply_transform(rot)
    
    #tri_mesh.export(out_fldr+"tf_mesh_out.obj")

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    
    scene = pyrender.Scene()
    scene.add(mesh)

    if plot_joints:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')
    parser.add_argument('--output-folder', required=False, type=str, default="./",                       
                        help='output folder for mesh and rendered image')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    output_folder = osp.expanduser(osp.expandvars(args.output_folder))

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         use_face_contour=use_face_contour, out_fldr=output_folder)
