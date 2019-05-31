from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np



## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 5
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = False


# Evaluation metric:
#metric = 'nll'  #or rmse
metric = 'rmse'

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/ha_lstm_05302019.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('/home/lei/workspace/data/trajectory/TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn) # 

lossVals = torch.zeros(5).cuda()
counts = torch.zeros(5).cuda()
lossVal = 0 # revised by Lei
count = 0

vehid = []
pred_x = []
pred_y = []
T = []
dsID = []
ts_cen = []
ts_nbr = []
wt_ha = []
nbr_location = []
print (len(tsDataloader.dataset) / 128)
for i, data in enumerate(tsDataloader):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds = data
    #if i == 2174:
    #    print (isinstance(hist, list))
    #a = list(hist.size()) # [16, 128, 2] if it is normal
    if not isinstance(hist, list): # nbrs are not zeros
        vehid.append(veh_id) # current vehicle to predict
    #print (veh_id.size())
        T.append(t) # current time
        dsID.append(ds)
    
        #print (i)
    # Initialize Variables
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        if metric == 'nll':
        # Forward pass
            if args['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, avg_along_time = True) # take average or not by Lei
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
        else:

            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            #print (fut_pred.size())
            l, c = maskedMSETest(fut_pred, fut, op_mask)

        fut_pred_x = fut_pred[:,:,0].detach()
        fut_pred_x = fut_pred_x.cpu().numpy()
        #print (type(fut_pred_x)) # (25, 128)
        fut_pred_y = fut_pred[:,:,1].detach()
        fut_pred_y = fut_pred_y.cpu().numpy()
        pred_x.append(fut_pred_x)
        pred_y.append(fut_pred_y)


    #print (len(pred))
        lossVal +=l.detach() # revised by Lei
        count += c.detach()


print ('lossVal is:', lossVal)
if metric == 'nll':
    print(lossVal / count)
else:
    print(torch.pow(lossVal / count,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters


