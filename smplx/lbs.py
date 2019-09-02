
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf       ##NOTE:facial landmarks not converted 
import numpy as np

'''
import torch
import torch.nn.functional as F
from .utils import rot_mat_to_euler

def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):
    

    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                 neck_kin_chain)
    rot_mats = batch_rodrigues(
        aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

    rel_rot_mat = torch.eye(3, device=vertices.device,
                            dtype=dtype).unsqueeze_(dim=0)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):
    

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor, vertices):
    
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    
    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
   
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_mat(R, t):
    
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

'''
### tensorflow converted part   #### FIGURE OUT HOW TO REPLACE TORCH.MATMUL  TORCH.BMM == TF.MATMUL

### face part here
def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=tf.float32):
    

    batch_size = vertices.shape[0]

    aa_pose = tf.gather(params=tf.reshape(pose,[batch_size, -1, 3]), axis=1, vertices=neck_kin_chain)
    rot_mats = tf.reshape(batch_rodrigues(tf.reshape(aa_pose,[-1, 3]), dtype=dtype), [batch_size, -1, 3, 3])

    rel_rot_mat = tf.expand_dims(tf.eye(3, dtype=dtype), axis=0)            #######.unsqueeze_(dim=0)   ### inplace version of unsqueeze
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = tf.matmul(rot_mats[:, idx], rel_rot_mat, name="bmm")

    y_rot_angle = tf.cast(tf.round(tf.clip_by_value(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, clip_value_min=dtype.min, clip_value_max=39)), dtype=tf.int64)
    neg_mask = tf.cast(tf.less(y_rot_angle, 0), dtype=tf.int64)
    mask = tf.cast(tf.less(y_rot_angle, -39), dtype=tf.int64)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = tf.gather(params=dynamic_lmk_faces_idx, axis=0, indices=y_rot_angle)
    dyn_lmk_b_coords = tf.gather(paramns=dynamic_lmk_b_coords, axis=0, indices=y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    #device = vertices.device

    lmk_faces = tf.reshape(tf.gather(params=faces, axis=0, indices=tf.reshape(lmk_faces_idx,[-1])), [batch_size, -1, 3])

    lmk_faces += tf.reshape(tf.convert_to_tensor(np.arange(int(batch_size), dtype=np.int64)),[-1, 1, 1]) * int(num_verts)

    temp = tf.gather(params= tf.reshape(vertices,[-1, 3]), axis=0, indices=lmk_faces)
    lmk_vertices = tf.reshape(temp, [batch_size, -1, 3, 3])

    landmarks = tf.einsum('blfi,blf->bli', lmk_vertices, lmk_bary_coords)
    return landmarks

##### face part end


def broadcastable_matmul(A, B, transpose_a=False, transpose_b=False):
    Andim = len(A.shape)
    Bndim = len(B.shape)
    if Andim == Bndim:
        return tf.matmul(A, B, transpose_a=transpose_a,
                         transpose_b=transpose_b)  
    with tf.name_scope('matmul'):
        a_index = Andim - (2 if transpose_a else 1)
        b_index = Bndim - (1 if transpose_b else 2)
        res = tf.tensordot(A, B, axes=[a_index, b_index])
        if Bndim > 2:               # only if B is batched, rearrange the axes
            A_Batch = np.arange(Andim - 2)
            M = len(A_Batch)
            B_Batch = (M + 1) + np.arange(Bndim - 2)
            N = (M + 1) + len(B_Batch)
            perm = np.concatenate((A_Batch, B_Batch, [M, N]))
            res = tf.transpose(res, perm)
    return tf.squeeze(res, axis=1)



def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=tf.float32):
    
    batch_size = max(betas.shape[0], pose.shape[0])    

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = tf.eye(3, dtype=dtype)
    if pose2rot:
        rot_mats = tf.reshape(batch_rodrigues(tf.reshape(pose, [-1, 3]), dtype=dtype), [batch_size, -1, 3, 3])

        pose_feature = tf.reshape((rot_mats[:, 1:, :, :] - ident), [batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = tf.reshape(broadcastable_matmul(pose_feature, posedirs), [batch_size, -1, 3])
    else:
        pose_feature = tf.reshape(pose[:, 1:],[batch_size, -1, 3, 3]) - ident
        rot_mats = tf.reshape(pose, [batch_size, -1, 3, 3])

        pose_offsets = tf.reshape(broadcastable_matmul(tf.reshape(pose_feature, [batch_size, -1]), posedirs), [batch_size, -1, 3])

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = tf.tile(tf.expand_dims(lbs_weights, axis=0), multiples=[batch_size, 1, 1])

    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = tf.reshape(broadcastable_matmul(W, tf.reshape(A,[batch_size, num_joints, 16])), [batch_size, -1, 4, 4])

    homogen_coord = tf.ones([batch_size, v_posed.shape[1], 1], dtype=dtype)

    v_posed_homo = tf.concat([v_posed, homogen_coord], axis=2)
    v_homo = broadcastable_matmul(T, tf.expand_dims(v_posed_homo, -1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed

def vertices2joints(J_regressor, vertices):    
    return tf.einsum('bik,ji->bjk', vertices, J_regressor)


def blend_shapes(betas, shape_disps):    
    blend_shape = tf.einsum('bl,mkl->bmk', betas, shape_disps)
    return blend_shape

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=tf.float32):
    batch_size = tf.shape(rot_vecs)[0]    

    angle = tf.norm(rot_vecs + 1e-8, axis=1, keep_dims=True)
    rot_dir = tf.div(rot_vecs, angle)

    angle = tf.expand_dims(angle, -1)
    cos = tf.cos(angle)
    sin = tf.sin(angle)

    # Bx1 arrays
    rx, ry, rz = tf.split(rot_dir, num_or_size_splits=3, axis=1)
    K = tf.zeros((batch_size, 3, 3), dtype=dtype)

    zeros = tf.zeros((batch_size, 1), dtype=dtype)
    K = tf.concat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
    K = tf.reshape(K,[batch_size, 3, 3])

    bmm = tf.matmul(K, K, name="bmm")
    ident = tf.expand_dims(tf.eye(3, dtype=dtype), 0)
    rot_mat = ident + sin * K + (1 - cos) * bmm
    return rot_mat

def transform_mat(R, t):    
    return tf.concat([tf.pad(R, [[0,0],[0,1],[0,0]]), tf.pad(t, [[0,0],[0,1],[0,0]], constant_values=1)], axis=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=tf.float32):
        

    joints = tf.expand_dims(joints, axis=-1)
    rel_joints = tf.Variable(joints)
    #rel_joints = tf.identity(joints)            ## torch.clone() replaced with tf.identity  ## verify
    #rel_joints[:, 1:] -= joints[:, parents[1:]]
    zer_shape = [joints.shape[0], 1, joints.shape[2], joints.shape[3]]
    
    rel_joints = rel_joints - tf.concat([tf.zeros(zer_shape, dtype=dtype), tf.gather(joints, parents[1:], axis=1)], axis=1)

    transforms_mat = tf.reshape(transform_mat(tf.reshape(rot_mats, [-1, 3, 3]), tf.reshape(rel_joints, [-1, 3, 1])), [-1, joints.shape[1], 4, 4])

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = broadcastable_matmul(tf.gather(params=transform_chain, indices=parents[i]), transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = tf.stack(transform_chain, axis=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = tf.pad(joints, [[0,0],[0,0],[0,1],[0,0]])

    rel_transforms = transforms - tf.pad(broadcastable_matmul(transforms, joints_homogen), [[0,0],[0,0],[0,0],[3,0]])

    return posed_joints, rel_transforms


####

