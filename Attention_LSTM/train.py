from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSE
from torch.utils.data import DataLoader
import time
import math
import datetime
import numpy as np
import random
import os


## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64 # lstm encoder hidden state size, adjustable
args['decoder_size'] = 128 # lstm decoder  hidden state size, adjustable
args['in_length'] = 16
args['out_length'] = 20
args['grid_size'] = (265,3) # (660/5 *2 +1 )*3
args['input_embedding_size'] = 32 # input dimension for lstm encoder, adjustable
args['train_flag'] = True




def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

def _init_fn(worker_id):
    np.random.seed(100 + worker_id)


best_val = 1000
start_time = datetime.datetime.now()

# Initialize network
net = highwayNet(args)
if args['use_cuda']:
    print ('cuda success!')
    net = net.cuda()


## Initialize optimizer
trainEpochs = 10
optimizer = torch.optim.Adam(net.parameters()) #lr = ...
batch_size = 128
crossEnt = torch.nn.BCELoss() # binary cross entropy

cav_ratio = 0.4 # change CAV percentage here -1 means CAV market place = 0, no neighbor CAV
t_h = 30 # historical step 30 
trSet = ngsimDataset('../../data/trajectory/TrainSet_us101.mat', t_h=t_h, CAV_ratio=cav_ratio)
valSet = ngsimDataset('../../data/trajectory/ValSet_us101.mat', t_h=t_h, CAV_ratio=cav_ratio)
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)


## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf



for epoch_num in range(trainEpochs):

    num_train = 0
    num_val = 0

## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

# Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0


    for i, data in enumerate(trDataloader):

        st_time = time.time()
        flag, hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds, targetID, targetLoc = data

        num_train += hist.size()[1]

        if flag == 0: # this happens when no target HDV in front
            continue
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()


        fut_pred, atten = net(hist, nbrs, mask, lat_enc, lon_enc)
        l = maskedMSE(fut_pred, fut, op_mask)

    # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)
    # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item() # sum mse for 100 batches
        avg_tr_time += batch_time

        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i) # average time/batch * rest batches
        # len(trset) total length; i * batch_size / len(trSet)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0 # clear the result every 100 batches
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
# _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False

    print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data  in enumerate(valDataloader):
        st_time = time.time()
        flag, hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds, targetID, targetLoc = data
    #print (hist)
    #print (nbrs)
    #print (mask)
    #print (flag)
        num_val += hist.size()[1]

        if flag == 0: # this happens when no target HDV in front
            continue

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

    # Forward pass


        fut_pred, atten = net(hist, nbrs, mask, lat_enc, lon_enc)

        l = maskedMSE(fut_pred, fut, op_mask)#maskedNLL(fut_pred, fut, op_mask)

        avg_val_loss += l.item()
        val_batch_count += 1

    print(avg_val_loss/val_batch_count)
    print ('num_train is', num_train)
    print ('num_num is', num_val)

# Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count

    if avg_val_loss/val_batch_count < best_val:
        print ('better model found!!!')
        torch.save(net.state_dict(), 'trained_models/front_529_lstm_horizon_40_att_cav_'+str(cav_ratio)+'_Hsteps_'+str(t_h) +'_whole.tar')
        best_val = avg_val_loss/val_batch_count

#torch.save(net.state_dict(), 'trained_models/ha_lstm_05122019_'+str(epoch_num)+'.tar')

end_time = datetime.datetime.now()

print('Total training time: ', end_time-start_time)
print('total training sample number:', num_train)
print('total val sample number:', num_val)
#__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

torch.save(net.state_dict(), 'trained_models/lstm_horizon_40_att_cav_'+str(cav_ratio)+'.tar')



