
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import tensorflow as tf
import numpy as np

from collections import namedtuple

#import torch
#import torch.nn as nn

from .lbs import (lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords)

from .vertex_ids import vertex_ids as VERTEX_IDS
from .utils import Struct, to_np, to_tensor
from .vertex_joint_selector import VertexJointSelector


ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'body_pose', 'expression',
                          'left_hand_pose', 'right_hand_pose',
                          'jaw_pose'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def create(model_path, model_type='smpl',
           **kwargs):
    
    # If it's a folder, assume
    if osp.isdir(model_path):
        model_path = os.path.join(model_path, model_type)

    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    elif model_type.lower() == 'smplh':
        return SMPLH(model_path, **kwargs)
    elif model_type.lower() == 'smplx':
        return SMPLX(model_path, **kwargs)
    #elif model_type.lower() == 'mano':
    #    return SMPLH_HANDS_ONLY(model_path, **kwargs)
    else:
        raise ValueError('Unknown model type {}, exiting!'.format(model_type))


class SMPL(object):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10

    def __init__(self, model_path, data_struct=None,
                 create_betas=True,
                 betas=None,
                 create_global_orient=True,
                 global_orient=None,
                 create_body_pose=True,
                 body_pose=None,
                 create_transl=True,
                 transl=None,
                 dtype=tf.float32,
                 batch_size=1,
                 joint_mapper=None, gender='neutral',
                 vertex_ids=None,
                 **kwargs):
      

        self.gender = gender

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
                smpl_path = os.path.join(model_path, model_fn)
            else:
                smpl_path = model_path
            assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
                smpl_path)

            with open(smpl_path, 'rb') as smpl_file:
                data_struct = Struct(**pickle.load(smpl_file,
                                                   encoding='latin1'))

        #super(SMPL, self).__init__()
        self.batch_size = batch_size

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.joint_mapper = joint_mapper

        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids, **kwargs)

        self.faces = data_struct.f
        self.faces_tensor = tf.Variable(to_tensor(to_np(self.faces, dtype=np.int64), dtype=tf.int64), name='faces_tensor', trainable=False)
        
        if create_betas:
            if betas is None:
                default_betas = tf.zeros([batch_size, self.NUM_BETAS], dtype=dtype)
            else:
                if 'tensorflow.python.framework.ops.Tensor' in str(type(betas)):
                    default_betas = betas #tf.identity(betas)#.detach()
                else:
                    default_betas = tf.convert_to_tensor(betas, dtype=dtype)

            self.betas = tf.Variable(default_betas, name='betas', trainable=True)

        # The tensor that contains the global rotation of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if create_global_orient:
            if global_orient is None:
                default_global_orient = tf.zeros([batch_size, 3], dtype=dtype)
            else:
                if 'tensorflow.python.framework.ops.Tensor' in str(type(global_orient)):
                    default_global_orient = global_orient #tf.identity(global_orient)#.detach()
                else:
                    default_global_orient = tf.convert_to_tensor(global_orient, dtype=dtype)

            global_orient = tf.Variable(default_global_orient, name='global_orient', trainable=True)
            self.global_orient = global_orient

        if create_body_pose:
            if body_pose is None:
                default_body_pose = tf.zeros( [batch_size, self.NUM_BODY_JOINTS * 3], dtype=dtype)
            else:
                if 'tensorflow.python.framework.ops.Tensor' in str(type(body_pose)):
                    default_body_pose = body_pose #tf.identity(body_pose)#.detach()
                else:
                    default_body_pose = tf.convert_to_tensor(body_pose,dtype=dtype)
            
            self.body_pose = tf.Variable(default_body_pose, name='body_pose', trainable=True)

        if create_transl:
            if transl is None:
                default_transl = tf.zeros([batch_size, 3], dtype=dtype)
            else:
                default_transl = tf.convert_to_tensor(transl, dtype=dtype)
            
            self.transl = tf.Variable(default_transl, name='transl', trainable=True)

        # The vertices of the template model
        self.v_template = tf.Variable(to_tensor(to_np(data_struct.v_template), dtype=dtype), name='v_template', trainable=False)

        # The shape components
        shapedirs = data_struct.shapedirs
        # The shape components
        self.shapedirs = tf.Variable( to_tensor(to_np(shapedirs), dtype=dtype), name='shapedirs', trainable=False)

        j_regressor = to_tensor(to_np( data_struct.J_regressor), dtype=dtype)
        self.J_regressor = tf.Variable(j_regressor, name='J_regressor', trainable=False)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(to_tensor(to_np(posedirs), dtype=dtype), name='posedirs', trainable=False)

        # indices of parents for each joints
        temp = to_np(data_struct.kintree_table[0])
        temp[0] = -1
        parents = tf.cast(to_tensor(temp), tf.int64)
        #parents[0] = -1
        self.parents = tf.Variable(parents, name='parents', trainable=False)

        self.lbs_weights = tf.Variable(to_tensor(to_np(data_struct.weights), dtype=dtype), name='lbs_weights', trainable=False)

    def create_mean_pose(self, data_struct):
        pass

    #@torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = tf.convert_to_tensor(params_dict[param_name])
            else:
                param = tf.fill(param.shape, 0)

    def get_num_verts(self):
        return self.v_template.shape[0]

    def get_num_faces(self):
        return self.faces.shape[0]

    def extra_repr(self):
        return 'Number of betas: {}'.format(self.NUM_BETAS)

    def __call__(self, betas=None, body_pose=None, global_orient=None,
                transl=None, return_verts=True, return_full_pose=False,
                **kwargs):
        
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = tf.concat([global_orient, body_pose], axis=1)

        if betas.shape[0] != self.batch_size:
            num_repeats = int(self.batch_size / betas.shape[0])
            betas = tf.tile(betas, multiples=[num_repeats, 1])           #.expand(num_repeats, -1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += tf.expand_dims(transl, axis=1)
            vertices += tf.expand_dims(transl, axis=1)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             joints=joints,
                             betas=self.betas,
                             full_pose=full_pose if return_full_pose else None)

        return output
'''
### added later
class SMPLH_HANDS_ONLY(SMPL):

    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS #2 * NUM_HAND_JOINTS

    def __init__(self, model_path,
                 data_struct=None,
                 data_structl=None,
                 data_structr=None,
                 create_left_hand_pose=True,
                 left_hand_pose=None,
                 create_right_hand_pose=True,
                 right_hand_pose=None,
                 use_pca=True,
                 num_pca_comps=6,
                 flat_hand_mean=False,
                 batch_size=1,
                 gender='male',
                 dtype=tf.float32,
                 vertex_ids=None,
                 use_compressed=True,
                 ext='pkl',
                 **kwargs):
        

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fnl = 'MANO_{}.{ext}'.format('LEFT', ext=ext)
                smplh_pathl = os.path.join(model_path, model_fnl)

                model_fnr = 'MANO_{}.{ext}'.format('RIGHT', ext=ext)
                smplh_pathr = os.path.join(model_path, model_fnr)
            else:
                smplh_pathl = model_pathl
                smplh_pathr = model_pathr

            assert osp.exists(smplh_pathl), 'Path {} does not exist!'.format(
                smplh_pathl)
            assert osp.exists(smplh_pathr), 'Path {} does not exist!'.format(
                smplh_pathr)

            if ext == 'pkl':
                with open(smplh_pathl, 'rb') as smplh_filel:
                    model_datal = pickle.load(smplh_filel, encoding='latin1')
                with open(smplh_pathr, 'rb') as smplh_filer:
                    model_datar = pickle.load(smplh_filer, encoding='latin1')
            elif ext == 'npz':
                model_datal = np.load(smplh_pathl, allow_pickle=True)
                model_datar = np.load(smplh_pathr, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            
            data_structl = Struct(**model_datal)
            data_structr = Struct(**model_datar)

            if(gender.upper() == 'LEFT'):           
                data_struct = data_structl
            else:            
                data_struct = data_structr

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']
       

        super(SMPLH_HANDS_ONLY, self).__init__(
                model_path=model_path, data_struct=data_struct,
                batch_size=batch_size, vertex_ids=vertex_ids, gender=gender,
                use_compressed=use_compressed, dtype=dtype, ext=ext, **kwargs)

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS

        
        left_hand_components = data_structl.hands_components[:num_pca_comps]
        self.np_left_hand_components = left_hand_components

        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(data_structl.hands_mean)
            right_hand_mean = np.zeros_like(data_structr.hands_mean)               
        else:
            left_hand_mean = data_structl.hands_mean   
            right_hand_mean = data_structr.hands_mean  
            
        if self.use_pca:
            self.left_hand_components = tf.Variable(tf.convert_to_tensor(left_hand_components, dtype=dtype), name='left_hand_components', trainable=False)

        self.left_hand_mean = tf.Variables(to_tensor(left_hand_mean, dtype=self.dtype), name='left_hand_mean',trainable=False)
            
        
        right_hand_components = data_structr.hands_components[:num_pca_comps]        
        self.np_right_hand_components = right_hand_components
        if self.use_pca:            
            self.right_hand_components = tf.Variable(name='right_hand_components', tf.convert_to_tensor(right_hand_components, dtype=dtype), trainable=False)        
            
            self.right_hand_mean = tf.Variable(to_tensor(right_hand_mean, dtype=self.dtype), name='right_hand_mean', trainable=False)
        

        if create_left_hand_pose:
            if left_hand_pose is None:
                default_lhand_pose = tf.zeros([batch_size, hand_pose_dim],
                                                 dtype=dtype)
            else:
                default_lhand_pose = tf.convert_to_tensor(left_hand_pose, dtype=dtype)

            left_hand_pose_param = tf.Variable(default_lhand_pose, name='left_hand_pose', trainable=True)
            self.left_hand_pose = left_hand_pose_param

        if create_right_hand_pose:
            if right_hand_pose is None:
                default_rhand_pose = tf.zeros([batch_size, hand_pose_dim],
                                                 dtype=dtype)
            else:
                default_rhand_pose = tf.convert_to_tensor(right_hand_pose, dtype=dtype)

            right_hand_pose_param = tf.Variable(default_rhand_pose, name='right_hand_pose', trainable=True)
            self.right_hand_pose = right_hand_pose_param
       
        
        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(data_struct, flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = tf.convert_to_tensor(pose_mean, dtype=dtype)
        self.rpose_mean = tf.Variable(pose_mean_tensor, name='pose_mean', trainable=False)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = tf.zeros([3], dtype=self.dtype)
        body_pose_mean = tf.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)

        pose_mean = tf.concat([global_orient_mean, body_pose_mean,
                               self.left_hand_mean,
                               self.right_hand_mean], axis=0)
        return pose_mean

    def extra_repr(self):
        msg = super(SMPLH_HANDS_ONLY, self).extra_repr()
        if self.use_pca:
            msg += '\nNumber of PCA components: {}'.format(self.num_pca_comps)
        msg += '\nFlat hand mean: {}'.format(self.flat_hand_mean)
        return msg

    def __call__(self, betas=None, global_orient=None, body_pose=None,
                left_hand_pose=None, right_hand_pose=None, transl=None,
                return_verts=True, return_full_pose=False,
                **kwargs):
        
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else self.right_hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        left_hand_pose = tf.einsum('bi,ij->bj', left_hand_pose, self.left_hand_components)
        right_hand_pose = tf.einsum('bi,ij->bj', right_hand_pose, self.right_hand_components)

        full_pose = tf.concat([global_orient, body_pose,
                               left_hand_pose,
                               right_hand_pose], axis=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(self.betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               dtype=self.dtype)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += tf.expand_dims(transl, axis=1)
            vertices += tf.expand_dims(transl, axis=1)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=self.betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             full_pose=full_pose if return_full_pose else None)

        return output
'''

class SMPLH(SMPL):

    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = SMPL.NUM_JOINTS - 2
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS

    def __init__(self, model_path,
                 data_struct=None,
                 create_left_hand_pose=True,
                 left_hand_pose=None,
                 create_right_hand_pose=True,
                 right_hand_pose=None,
                 use_pca=True,
                 num_pca_comps=6,
                 flat_hand_mean=False,
                 batch_size=1,
                 gender='neutral',
                 dtype=tf.float32,
                 vertex_ids=None,
                 use_compressed=True,
                 ext='pkl',
                 **kwargs):
        

        self.num_pca_comps = num_pca_comps
        # If no data structure is passed, then load the data from the given
        # model folder
        if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'SMPLH_{}.{ext}'.format(gender.upper(), ext=ext)
                smplh_path = os.path.join(model_path, model_fn)
            else:
                smplh_path = model_path
            assert osp.exists(smplh_path), 'Path {} does not exist!'.format(
                smplh_path)

            if ext == 'pkl':
                with open(smplh_path, 'rb') as smplh_file:
                    model_data = pickle.load(smplh_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(smplh_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)

        if vertex_ids is None:
            vertex_ids = VERTEX_IDS['smplh']

        super(SMPLH, self).__init__(
            model_path=model_path, data_struct=data_struct,
            batch_size=batch_size, vertex_ids=vertex_ids, gender=gender,
            use_compressed=use_compressed, dtype=dtype, ext=ext, **kwargs)

        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        left_hand_components = data_struct.hands_componentsl[:num_pca_comps]
        right_hand_components = data_struct.hands_componentsr[:num_pca_comps]

        self.np_left_hand_components = left_hand_components
        self.np_right_hand_components = right_hand_components
        if self.use_pca:
            self.left_hand_components = tf.Variable(tf.convert_to_tensor(left_hand_components, dtype=dtype), name='left_hand_components', trainable=False)
            self.right_hand_components = tf.Variable(tf.convert_to_tensor(right_hand_components, dtype=dtype), name='right_hand_components', trainable=False)

        if self.flat_hand_mean:
            left_hand_mean = np.zeros_like(data_struct.hands_meanl)
        else:
            left_hand_mean = data_struct.hands_meanl

        if self.flat_hand_mean:
            right_hand_mean = np.zeros_like(data_struct.hands_meanr)
        else:
            right_hand_mean = data_struct.hands_meanr

        self.left_hand_mean = tf.Variable(to_tensor(left_hand_mean, dtype=self.dtype), name='left_hand_mean', trainable=False)
        self.right_hand_mean = tf.Variable(to_tensor(right_hand_mean, dtype=self.dtype), name='right_hand_mean', trainable=False)

        # Create the buffers for the pose of the left hand
        hand_pose_dim = num_pca_comps if use_pca else 3 * self.NUM_HAND_JOINTS
        if create_left_hand_pose:
            if left_hand_pose is None:
                default_lhand_pose = tf.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_lhand_pose = tf.convert_to_tensor(left_hand_pose, dtype=dtype)

            left_hand_pose_param = tf.Variable(default_lhand_pose, name='left_hand_pose', trainable=True)
            self.left_hand_pose = left_hand_pose_param
            
        if create_right_hand_pose:
            if right_hand_pose is None:
                default_rhand_pose = tf.zeros([batch_size, hand_pose_dim], dtype=dtype)
            else:
                default_rhand_pose = tf.convert_to_tensor(right_hand_pose, dtype=dtype)

            right_hand_pose_param = tf.Variable(default_rhand_pose, name='right_hand_pose', trainable=True)
            self.right_hand_pose = right_hand_pose_param

        # Create the buffer for the mean pose.
        pose_mean = self.create_mean_pose(data_struct, flat_hand_mean=flat_hand_mean)
        pose_mean_tensor = tf.convert_to_tensor(pose_mean, dtype=dtype)
        self.pose_mean = tf.Variable(pose_mean_tensor, name='pose_mean', trainable=False)

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = tf.zeros([3], dtype=self.dtype)
        body_pose_mean = tf.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)

        pose_mean = tf.concat([global_orient_mean, body_pose_mean,
                               self.left_hand_mean,
                               self.right_hand_mean], axis=0)
        return pose_mean

    def extra_repr(self):
        msg = super(SMPLH, self).extra_repr()
        if self.use_pca:
            msg += '\nNumber of PCA components: {}'.format(self.num_pca_comps)
        msg += '\nFlat hand mean: {}'.format(self.flat_hand_mean)
        return msg

    def __call__(self, betas=None, global_orient=None, body_pose=None,
                left_hand_pose=None, right_hand_pose=None, transl=None,
                return_verts=True, return_full_pose=False,
                **kwargs):
        
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        left_hand_pose = (left_hand_pose if left_hand_pose is not None else self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else self.right_hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        left_hand_pose = tf.einsum('bi,ij->bj', left_hand_pose, self.left_hand_components)
        right_hand_pose = tf.einsum('bi,ij->bj', right_hand_pose, self.right_hand_components)

        full_pose = tf.concat([global_orient, body_pose,
                               left_hand_pose,
                               right_hand_pose], axis=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(self.betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               dtype=self.dtype)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += tf.expand_dims(transl, axis=1)
            vertices += tf.expand_dims(transl, axis=1)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=self.betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             left_hand_pose=left_hand_pose,
                             right_hand_pose=right_hand_pose,
                             full_pose=full_pose if return_full_pose else None)

        return output


class SMPLX(SMPLH):
    

    NUM_BODY_JOINTS = SMPLH.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    NUM_EXPR_COEFFS = 10
    NECK_IDX = 12

    def __init__(self, model_path,
                 create_expression=True, expression=None,
                 create_jaw_pose=True, jaw_pose=None,
                 create_leye_pose=True, leye_pose=None,
                 create_reye_pose=True, reye_pose=None,
                 use_face_contour=False,
                 batch_size=1, gender='neutral',
                 dtype=tf.float32,
                 ext='npz',
                 **kwargs):
       

        # Load the model
        if osp.isdir(model_path):
            model_fn = 'SMPLX_{}.{ext}'.format(gender.upper(), ext=ext)
            smplx_path = os.path.join(model_path, model_fn)
        else:
            smplx_path = model_path
        assert osp.exists(smplx_path), 'Path {} does not exist!'.format(
            smplx_path)

        if ext == 'pkl':
            with open(smplx_path, 'rb') as smplx_file:
                model_data = pickle.load(smplx_file, encoding='latin1')
        elif ext == 'npz':
            model_data = np.load(smplx_path, allow_pickle=True)
        else:
            raise ValueError('Unknown extension: {}'.format(ext))

        data_struct = Struct(**model_data)

        super(SMPLX, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            vertex_ids=VERTEX_IDS['smplx'],
            gender=gender, ext=ext,
            **kwargs)

        lmk_faces_idx = data_struct.lmk_faces_idx
        self.lmk_faces_idx = tf.Variable(tf.convert_to_tensor(lmk_faces_idx, dtype=tf.int64), name='lmk_faces_idx', trainable=False)
        lmk_bary_coords = data_struct.lmk_bary_coords
        self.lmk_bary_coords = tf.Variable(tf.convert_to_tensor(lmk_bary_coords, dtype=dtype), name='lmk_bary_coords', trainable=False)

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            dynamic_lmk_faces_idx = data_struct.dynamic_lmk_faces_idx
            dynamic_lmk_faces_idx = tf.convert_to_tensor( dynamic_lmk_faces_idx,dtype=tf.int64)
            self.dynamic_lmk_faces_idx = tf.Variable(dynamic_lmk_faces_idx, name='dynamic_lmk_faces_idx', trainable=False)

            dynamic_lmk_bary_coords = data_struct.dynamic_lmk_bary_coords
            dynamic_lmk_bary_coords = tf.convert_to_tensor( dynamic_lmk_bary_coords, dtype=dtype)
            self.dynamic_lmk_bary_coords = tf.Variable(dynamic_lmk_bary_coords, name='dynamic_lmk_bary_coords', trainable=False)

            neck_kin_chain = []
            curr_idx = tf.convert_to_tensor(self.NECK_IDX, dtype=tf.int64)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.neck_kin_chain = tf.Variable(tf.stack(neck_kin_chain), name='neck_kin_chain', trainable=False)

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = tf.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = tf.convert_to_tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = tf.Variable(default_jaw_pose, name='jaw_pose', trainable=True)
            self.jaw_pose = jaw_pose_param

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = tf.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = tf.convert_to_tensor(leye_pose, dtype=dtype)
            leye_pose_param = tf.Variable(default_leye_pose, name='leye_pose', trainable=True)
            self.leye_pose = leye_pose_param

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = tf.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = tf.convert_to_tensor(reye_pose, dtype=dtype)
            reye_pose_param = tf.Variable(default_reye_pose, name='reye_pose', trainable=True)
            self.reye_pose = reye_pose_param

        if create_expression:
            if expression is None:
                default_expression = tf.zeros( [batch_size, self.NUM_EXPR_COEFFS], dtype=dtype)
            else:
                default_expression = tf.convert_to_tensor(expression, dtype=dtype)
            expression_param = tf.Variable(default_expression, name='expression', trainable=True)
            self.expression = expression_param

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = tf.zeros([3], dtype=self.dtype)
        body_pose_mean = tf.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)
        jaw_pose_mean = tf.zeros([3], dtype=self.dtype)
        leye_pose_mean = tf.zeros([3], dtype=self.dtype)
        reye_pose_mean = tf.zeros([3], dtype=self.dtype)

        pose_mean = tf.concat([global_orient_mean, body_pose_mean,
                                    jaw_pose_mean,
                                    leye_pose_mean, reye_pose_mean,
                                    self.left_hand_mean, self.right_hand_mean],
                                    axis=0)

        return pose_mean

    def extra_repr(self):
        msg = super(SMPLX, self).extra_repr()
        msg += '\nGender: {}'.format(self.gender.title())
        msg += '\nExpression Coefficients: {}'.format(
            self.NUM_EXPR_COEFFS)
        msg += '\nUse face contour: {}'.format(self.use_face_contour)
        return msg

    def __call__(self, betas=None, global_orient=None, body_pose=None,
                left_hand_pose=None, right_hand_pose=None, transl=None,
                expression=None, jaw_pose=None, leye_pose=None, reye_pose=None,
                return_verts=True, return_full_pose=False, **kwargs):
        

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        left_hand_pose = (left_hand_pose if left_hand_pose is not None else self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose
        expression = expression if expression is not None else self.expression

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        left_hand_pose = tf.einsum('bi,ij->bj', left_hand_pose, self.left_hand_components)
        right_hand_pose = tf.einsum('bi,ij->bj', right_hand_pose, self.right_hand_components)

        full_pose = tf.concat([global_orient, body_pose,
                               jaw_pose, leye_pose, reye_pose,
                               left_hand_pose,
                               right_hand_pose], axis=1)

        # Add the mean pose of the model. Does not affect the body, only the
        # hands when flat_hand_mean == False
        full_pose += self.pose_mean

        batch_size = max(betas.shape[0], global_orient.shape[0], body_pose.shape[0])
        # Concatenate the shape and expression coefficients
        shape_components = tf.concat([tf.tile(betas, multiples=[int(int(batch_size) / int(betas.shape[0])), 1]), expression], axis=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               dtype=self.dtype)

        lmk_faces_idx = tf.expand_dims(self.lmk_faces_idx, axis=0)
        lmk_bary_coords = tf.tile(tf.expand_dims(self.lmk_bary_coords,axis=0), multiples=[self.batch_size, 1, 1])
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = tf.concat([tf.tiles(lmk_faces_idx, multiples=[batch_size, 1]), dyn_lmk_faces_idx], 1)
            lmk_bary_coords = tf.concat([tf.tile(lmk_bary_coords, multiples=[batch_size, 1, 1]), dyn_lmk_bary_coords], 1)      ### can cause possible shape problem

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = tf.concat([joints, landmarks], axis=1)
        # Map the joints to the current dataset

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += tf.expand_dims(transl, axis=1)
            vertices += tf.expand_dims(transl, axis=1)

        output = ModelOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=self.global_orient,
                             body_pose=body_pose,
                             left_hand_pose=self.left_hand_pose,
                             right_hand_pose=self.right_hand_pose,
                             jaw_pose=jaw_pose,
                             full_pose=full_pose if return_full_pose else None)
        return output
