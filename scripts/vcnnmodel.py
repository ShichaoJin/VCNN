import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from IPython import embed



class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.ReLU):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = activation_func()
        # with learnable parameters
        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.

    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels, kernel_size)
        self.conv_2 = conv3d(out_channels, out_channels, kernel_size)
        self.conv_3 = conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_1 + z_3


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU): # 256, 256
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, kernel_size, stride)
        self.lhs_conv = conv3d(out_channels // 2, out_channels, kernel_size)
        self.conv_x3 = conv3d_x3(out_channels, out_channels, kernel_size)

    def forward(self, lhs, rhs):  # 128 ....256
        #print(lhs.size(), rhs.size())
        rhs_up = self.up(rhs) # 256>>> 256 , but grain doubled
        lhs_conv = self.lhs_conv(lhs) #128 >>> 256
        rhs_add = crop(rhs_up, lhs_conv) + lhs_conv  #element-wise add
        return self.conv_x3(rhs_add)


def crop(large, small):
    """large / small with shape [batch_size, channels, depth, height, width]"""

    l, s = large.size(), small.size()
    offset = [0, 0, (l[2] - s[2]) // 2, (l[3] - s[3]) // 2, (l[4] - s[4]) // 2]
    return large[..., offset[2]: offset[2] + s[2], offset[3]: offset[3] + s[3], offset[4]: offset[4] + s[4]]


def conv3d_as_pool(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        activation_func())


def deconv3d_as_up(in_channels, out_channels, kernel_size=3, stride=2, activation_func=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        activation_func()
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels, criterion):
        super(softmax_out, self).__init__()
        self._K = out_channels
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        if criterion == 'nll':
            self.softmax = F.log_softmax
        else:
            assert criterion == 'dice', "Expect `dice` (dice loss) or `nll` (negative log likelihood loss)."
            self.softmax = F.softmax

    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        # Put channel axis in the last dim for softmax.
        y_perm = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        y_flat = y_perm.view(-1, self._K)
        return self.softmax(y_flat),y_conv



class conv12(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv12, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        # Put channel axis in the last dim for softmax.
        rs_fts = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        

        return rs_fts




#--- crop3D segmentation
def disc_loss(fts, y, conf):
    N,C = fts.size()
    y = y.contiguous()          #actual y
    #--
    try:
        num = y.max().data[0] + 1
    except:
        num = y.max().item() + 1
    rs_var, rs_dist, rs_center = [], [], []
    cc = []
    for k in range(num):
        msk = y.eq(k).view(-1,1).expand_as(fts)
        val = fts.masked_select(msk).view(-1,C)
        cur_center = val.mean(0)
        rs_center.append(cur_center)
        #--
        val = torch.norm(val-cur_center.expand_as(val),2,1)   #may be use L1 distance,
        cc.append(val)
        val = val-conf['c0']
        val = torch.clamp(val,min=1.e-7)
        val = val*val                            #2017/8/19     
        #
        rs_var.append(val.mean()) 
    '''
    for i in cc:
        tmp =  i.data.cpu().numpy().mean()
        mean.append(tmp)
    '''  
    #--
    dd =[]
    for i in range(1,num):          # from 1, ingore noise
        for j in range(num):
            if i>=j: continue
            dd.append((rs_center[i]-rs_center[j]).norm(p=2))
            val = 2*conf['c1']-(rs_center[i]-rs_center[j]).norm(p=2) #may be use L1 distance,
            val = torch.clamp(val, min=1.e-7)
            val = val*val                               #2017/8/19 
            rs_dist.append(val)
    #--

    rs_center_ = torch.stack(rs_center).view(-1,C)
    rs_center = torch.norm(rs_center_,2,1)
    rs_center = torch.clamp(rs_center, min=1.e-7)

    #--2017/8/11
    #print('line 203'),embed()
    rs_add = torch.norm(rs_center-6*conf['c0'],2,1)   #may be use L1 distance,
    rs_add = torch.clamp(rs_add, min=1.e-7)
    #--
    a,b,c,d = conf['abcd']
    #loss = a*torch.stack(rs_var).mean() + b*torch.stack(rs_dist).mean() + c*rs_center.mean()+d*rs_add.mean() -d*6*conf['c0']#
    loss = a*torch.stack(rs_var).mean() + b*torch.stack(rs_dist).mean() + d*rs_add.mean() 
    return loss

class VCNN(nn.Module):
    def __init__(self, K, criterion):
        super(VCNN, self).__init__()
        self.conv_1 = conv3d_x3(1, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x3(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)

        self.bottom = conv3d_x3(256, 256)

        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x3(128, 64)
        self.deconv_1 = deconv3d_x3(64, 32)

        self.out = softmax_out(32, K, criterion)
        self.out2 = conv12(32+K,32)              #2017/8/21
        self.K = K

    def forward(self, x):
        #print('vnet --'),embed()

        conv_1 = self.conv_1(x)
        #np.savez("D:/conv1.npz",arr=conv_1.data.cpu().numpy())
        pool = self.pool_1(conv_1)
        #np.savez("D:/pool1.npz",arr=pool.data.cpu().numpy())

        conv_2 = self.conv_2(pool)
        #np.savez("D:/conv2.npz",arr=conv_2.data.cpu().numpy())
        pool = self.pool_2(conv_2)
        #np.savez("D:/pool2.npz",arr=pool.data.cpu().numpy())

        conv_3 = self.conv_3(pool)
        #np.savez("D:/conv3.npz",arr=conv_3.data.cpu().numpy())
        pool = self.pool_3(conv_3)
        #np.savez("D:/pool3.npz",arr=pool.data.cpu().numpy())

        conv_4 = self.conv_4(pool)
        #np.savez("D:/conv4.npz",arr=conv_4.data.cpu().numpy())
        pool = self.pool_4(conv_4)
        #np.savez("D:/pool4.npz",arr=pool.data.cpu().numpy())

        bottom = self.bottom(pool)
        #np.savez("D:/bottom.npz",arr=bottom.data.cpu().numpy())

        deconv = self.deconv_4(conv_4, bottom)
        #np.savez("D:/deconv4.npz",arr=deconv.data.cpu().numpy())


        deconv = self.deconv_3(conv_3, deconv)
        #np.savez("D:/deconv3.npz",arr=deconv.data.cpu().numpy())

        deconv = self.deconv_2(conv_2, deconv)
        #np.savez("D:/deconv2.npz",arr=deconv.data.cpu().numpy())

        deconv = self.deconv_1(conv_1, deconv)
        #np.savez("D:/deconv1.npz",arr=deconv.data.cpu().numpy())

        #rs_cls = self.out(deconv)
        rs_cls,y_conv = self.out(deconv)  
        #np.savez("D:/cls2.npz",arr=y_conv.data.cpu().numpy())

        #may better if we select y_conv and deconv here,before cat, using the rs_cls index_select(0,label[0,...,-1])
        fts = torch.cat([y_conv,deconv], dim=1)   #chanal ==34
        assert fts.size(1)==32+self.K
        rs_fts = self.out2(fts)  # two convolution layer...

        #np.savez("D:/segfts.npz",arr=rs_fts.data.cpu().numpy())

        rs_fts = rs_fts.view(rs_cls.size()[0], -1)
        
        
        return rs_cls, rs_fts
   

