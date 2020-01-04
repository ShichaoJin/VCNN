from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
#import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json
from scipy.sparse import csr_matrix
from h5 import *
from IPython import embed
np.random.seed(1234567)
torch.manual_seed(1234567)
def rot_c3x3(rpy):
    cx,cy,cz = np.cos(rpy)     # Q: roll,pitch,yaw| trans.
    sx,sy,sz = np.sin(rpy)
    
    return np.array([          # R*x => y
        [cy*cz, cz*sx*sy-cx*sz, cx*cz*sy+sx*sz,],
        [cy*sz, cx*cz+sx*sy*sz, cx*sy*sz-cz*sx,],
        [-sy,   cy*sx,          cx*cy,]
    ])

class PartDataset(data.Dataset):
    def __init__(self, fname, size, batch, rot_limit=None, trans_limit=None,  train=True):
        self._fname = fname
        assert os.path.exists(fname)
        self._xyz = None
        with h5f_(fname, mode='r') as h5f:
        #if 1:
            x =h5f.get_node('/process/x')
            y1 = h5f.get_node('/process/y1').read()
            y2 = h5f.get_node('/process/y2').read()
            idxs = h5f.get_node('/process/idxs').read()

            embed()
            self._cls = y1
            self._parts = y2
            #self._fidx = np.array([(0,self._xyz.shape[0]),]) 
            self._fidx = idxs                          
            self._size = size
            self._batch = batch
            self._itr = -1
            self._rot_limit   = rot_limit if not rot_limit is None else np.r_[5., 5., 360.]*np.pi/180 
            self._trans_limit = trans_limit if not trans_limit is None else np.r_[0., 0., 0.] 
            #--
            self._nset = self._fidx.shape[0]
            self._train = train

    def _get_data(self, i):
        ran = np.random.rand(6)*2-1
        rot     = ran[:3]*self._rot_limit     
        trans   = ran[3:]*self._trans_limit         #   5m
        #-- rand choice pts        
        s = slice(*self._fidx[i])
        cur_xyz = self._xyz[s]   #-- shuffled
        cur_xyz_o = np.copy(cur_xyz)
        cur_cls = self._cls[s].astype('i4')         
        cur_part = self._parts[s].astype('i4')
        #K = (1<<17)
        #cur_xyz = cur_xyz[:K,...]
        #cur_cls = cur_cls[:K].reshape(1,-1)
        cur_xyz = cur_xyz.dot(rot_c3x3(rot)) #+ trans
        #cur_xyz = cur_xyz.transpose(1,0).astype('f4')             # 3 x N
        
        #-- scale
        cur_xyz -= cur_xyz.min(0)
        scale = 31/cur_xyz.max(0).min()
        cur_xyz = (cur_xyz*scale).astype('i4')
        #-- unique
        _val = cur_xyz.view('S12').flatten()
        _sorted = _val.argsort()
        _, keep = np.unique(_val[_sorted], return_index=True)
        keep = _sorted[keep]
        
        cur_xyz = cur_xyz[keep,...]
        cur_cls = cur_cls[keep]
        cur_part = cur_part[keep]

        #-- convert tor volumn
        nx, ny, nz = cur_xyz.max(0)+1
        val = np.ones(cur_xyz.shape[0], dtype='f4')
        idx = (cur_xyz[:,0]*ny+cur_xyz[:,1], cur_xyz[:,2])
        gt_x = csr_matrix( (val, idx), shape=(nx*ny, nz))
        gt_x = gt_x.astype('f4').toarray().reshape(1,nx,ny,nz)
        
        roi = (cur_xyz[:,0]*ny+cur_xyz[:,1])*nz + cur_xyz[:,2]
        gt_y = np.c_[cur_cls, cur_part, roi].astype('i8')      # y|roi
        return (gt_x, gt_y,cur_xyz,scale,cur_xyz_o) 



    def __getitem__(self, index):
        
        if index%self._batch==0:       # fix itr for dbg
            self._itr = (self._itr+1)%self._nset

        if self._xyz is None:
            _h5f = h5f_(self._fname, mode='r')
            self._xyz = _h5f.get_node('/process/x')

        if self._train == True:
            xyz, cls,cur_xyz,scale ,cur_xyz_o= self._get_data(self._itr)
            xyz = torch.from_numpy(xyz)
            cls = torch.from_numpy(cls)
            return (xyz, cls,cur_xyz,scale,cur_xyz_o)    #image, label,cur_xyz,scale,cur_xyz_o

        else:
            xyz, cls,cur_xyz,scale ,minxyz,cur_xyz_o,cur_cls_label,cur_part_label= self._get_test_data(self._itr)
            xyz = torch.from_numpy(xyz)
            cls = torch.from_numpy(cls)
            return (xyz, cls,cur_xyz,scale,minxyz,cur_xyz_o,cur_cls_label,cur_part_label)


    def __len__(self):
        return self._size


if __name__ == '__main__':
    #from greenvalley.core import *
    #from _core import *
    from h5 import *

    root = os.path.dirname(__file__) 
    root = os.path.abspath('../../%s'%root)
    print(root)

    d = PartDataset('%s/x_y1_y2_idx_2277_withoutnoise.h5'%root, 4, 2)

    print('line 189'),embed()
    print(len(d))
    ps, seg = d[0]
    
    embed()
