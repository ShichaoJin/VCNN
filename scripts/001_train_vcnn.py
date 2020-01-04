#!coding:gbk
# The code is written by Shichao Jin (金时超) from University of Chinese Academy of Sciences. The code comments are hybrid with English and Chinese.
import os,logging
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vcnnmodel import VCNN,disc_loss
from utils import *
from datasets import PartDataset
from scipy.spatial import cKDTree
#from sklearn.manifold import TSNE   #for feature visulization
from IPython import embed

root = os.path.abspath('../%s'% os.path.dirname(__file__) )  ##  locate to your directory:  ../../VCNN 
# Set net configuration
conf = config()
# GPU configuration
def init():
    conf.prefix = 'vcnn'
    conf.checkpoint_dir += conf.prefix
    conf.result_dir += conf.prefix
    conf.learning_rate = 1e-5  #-5
    conf.from_scratch = True #False
    conf.resume_step = -1
    conf.criterion = 'nll'  # 'dice' or 'nll'
    conf.batch_size = 1
    conf.epochs = 1000
    #conf.threads = 1
    conf.threads = 0
    conf.num_classes = 2
    conf.model = '' #'%s/data/model/model_32x_mean_add_v2277_50.pth'%root
    conf.disc_loss_conf = {
        'c0':   1,
        'c1':   3,
        'abcd':  [1,1, 0.01,0.01],
    }
    conf._pid = os.getpid() 
    if conf.cuda:
        torch.cuda.set_device(0)
        print('===> Current GPU device is', torch.cuda.current_device())
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)

def training_data_loader():
    #_fname = '%s/data/x_y1_y2_idx_2017060708_normalise.h5'%root  
    _fname = '%s/data/x_y1_y2_idx_33_withoutnoise.h5'%root  
    dataset = PartDataset(_fname,300, conf.batch_size,train=True)  #train=True  3050
    return DataLoader(dataset, batch_size=conf.batch_size,
                     shuffle=False, num_workers=conf.threads)
def validation_data_loader():
    #_fname = '%s/data/x_y1_y2_idx_200_withoutnoise.h5'%root
    _fname = '%s/data/x_y1_y2_idx_33_withoutnoise.h5'%root
    dataset = PartDataset(_fname,1, conf.batch_size,train=False)  #train=False
    return DataLoader(dataset, batch_size=conf.batch_size,
                     shuffle=False, num_workers=conf.threads)

#-----------------------------------------------------------------------main
def main(resume_id, model_name):
    print('===> Building vnet...')
    np.random.seed(123456)
    init()
    model_name = '%s/data/model/%s_%%03d.pth'%(root, model_name,)
    n=2                                                           #save sequence
    if resume_id>0:
        conf.model = model_name%(resume_id-n) #2017/8/17
    model = VCNN(conf.num_classes, conf.criterion)
    if os.path.exists(conf.model):
        print ('... load %s'%conf.model)
        model.load_state_dict(torch.load(conf.model))
    else:
        print('initial-------------new weights-----')
        model.apply(weights_init)
    if conf.cuda:
        model = model.cuda()
    print('===> Loss function: {}'.format(conf.criterion))
    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    start_i = resume_id
    # Define optimizer, loss is related to conf.criterion.
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)   #2018/8/22 1
    #optimizer = optim.SGD(model.parameters(), lr = conf.learning_rate, momentum=0.9)  ##if grad explore, then comment the optimizer

    total_i = conf.epochs * 1 #conf.augment_size
    cc  = conf.disc_loss_conf 
    R = cc['c0']

    def train():
        epoch_loss = 0
        epoch_overlap = 0
        epoch_cluster = 0
        epoch_acc = 0
        # Sets the module in training mode.This has any effect only on modules such as Dropout or BatchNorm.
        model.train()

        for partial_epoch, (image, label,cur_xyz,scale,cur_xyz_o) in enumerate(training_data_loader(), 0):
            #image  torch.Size([1, 1, 93, 32, 67]) is the size of voxels in x-y-z directions. 是体素的格子
            #label  torch.Size([1, 1631, 3]) three columns[classification label, segmentation label, index]; 三列，分别为分类的label, 分割的label，位置索引
            #cur_xyz  torch.Size([1, 1631, 3]) voxels; 为有目标的体素单元
            #cur_xyz_o  torch.Size([1, 5809, 3]) raw points; 为所有的原始点云
            #scale represents the normalized scale 缩放比列
            _target = label.numpy()[0,...,0].flatten()
            w0 = _target[_target==0].shape[0]
            w1 = _target[_target==1].shape[0]
            _w0 = float(w1)/(w0+w1)
            _w1 = float(w0)/(w0+w1)
            w0 = _w0/(_w0+_w1)
            w1 = _w1/(_w0+_w1)
            w = torch.from_numpy(np.array([w0,w1]).astype('f4')).cuda()
            image, label = Variable(image,requires_grad=False).float(), Variable(label,requires_grad=False).long()
            if conf.cuda:
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad() 


            output,fts = model(image)

            output = output.contiguous().index_select(0,label[0,...,-1])
            fts = fts.contiguous().index_select(0,label[0,...,-1])
            p= cur_xyz.numpy()[0]

            cls_loss = F.nll_loss(output, label[0,...,0], weight=w)
            part_loss = disc_loss(fts, label[0,...,1], cc)
            loss = cls_loss + 1*part_loss
            loss.backward()   
            optimizer.step() 

            for pa in model.parameters():
                if pa.grad is None:continue
                pa.grad.data = pa.grad.data.clamp(-1,1)
                pa.data.add_(-conf.learning_rate, pa.grad.data)

            epoch_loss += loss.data[0]
            # Compute dice overlap by `argmax`
            pred = output.data.max(1)[1]
            true = label[0,...,0].data.long()
            dice_overlap = 2 * torch.sum(pred * true) / (torch.sum(pred) + torch.sum(true)) * 100
            epoch_overlap += dice_overlap

            epoch_cluster += part_loss.data[0]

            # Compute accuracy
            accuracy = pred.eq(true).cpu().sum() / true.numel() * 100
            epoch_acc += accuracy
            
            if partial_epoch%100==0:
                print('... i|%s acc|%.2f%%'%(partial_epoch, accuracy))

        avg_loss, avg_dice, avg_acc, avg_clustering = np.array([epoch_loss, epoch_overlap, epoch_acc, epoch_cluster]) / (partial_epoch+1)
        print_format = [i, i // conf.augment_size + 1, conf.epochs, avg_loss, avg_dice, avg_acc, avg_clustering]
        print(
            '===>{} ({}/{})\tLoss: {:.5f}\tDice Overlap {:.2f}%\tAccuracy: {:.2f}%  clustering|{:.5f}'.format(*print_format))
        print('learning rate:----',conf.learning_rate)
        with open('%s/data/model/loss.txt'%root,'a+') as f:
            f.write('i:%d, avg_loss: %s, avg_dice: %s, avg_acc: %s, avg_clustering: %s,learning rate:%s  \n'%(i,avg_loss, avg_dice, avg_acc, avg_clustering,conf.learning_rate ))
        image = label = cur_xyz = scale = None

        return avg_loss, avg_dice, avg_acc, avg_clustering
    

    for i in range(start_i, total_i + 1):
        train_results = train()

        if  (i+1) % n == 0:
            torch.save(model.state_dict(), model_name%i)          
    
        if (i+1)%20 ==0 and (i+1)<=100:
            conf.learning_rate = conf.learning_rate/(np.sqrt(i+1))#2.0   #1.e-4 ----1.99.e-7

        
if __name__ == '__main__':
    import argparse
    import  logging
    logging.basicConfig(format="%(asctime)s %(levelname)-8s | %(message)s",level=logging.DEBUG, datefmt='%m-%d %H:%M')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='training from <resume id>')
    parser.add_argument('--model', default='model_v0', help='model_name_<id>')
    args = parser.parse_args()

    main(args.resume, args.model)
