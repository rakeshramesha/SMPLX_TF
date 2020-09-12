
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from .utils import to_tensor

class VertexJointSelector(object):

    def __init__(self, vertex_ids=None, use_hands=True, use_feet_keypoints=True, **kwargs):
        #super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs, face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate([extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate([extra_joints_idxs, tips_idxs])
        
        self.extra_joints_idxs = tf.Variable(to_tensor(extra_joints_idxs, dtype=tf.int64), name='extra_joints_idxs', trainable=False)   

    def __call__(self, vertices, joints):
        extra_joints = tf.gather(params=vertices, indices=self.extra_joints_idxs, axis=1)
        joints = tf.concat([joints, extra_joints], axis=1)

        return joints
