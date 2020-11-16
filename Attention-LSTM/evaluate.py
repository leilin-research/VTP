from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSETest, maskedMAETest
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np



## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128#64
args['in_length'] = 16
args['out_length'] = 20
args['grid_size'] = (265,3)

args['input_embedding_size'] = 32

args['train_flag'] = False
args['dropout'] = 0
batch_size = 128

# Evaluation metric:
#metric = 'nll'  #or rmse
metric = 'rmse'
cav = 0.4 # set as -1, when no cavs at all
t_h = 30 # 30 or 60
r = 3
# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/lstm_horizon_40_att_cav_'+str(cav)+'.tar')) 
#net.load_state_dict(torch.load('trained_models/front_529_lstm_horizon_40_att_cav_'+str(cav)+'_Hsteps_'+str(t_h)+'_whole_round'+str(r+1)+'.tar')) #
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('../../data/trajectory/TestSet_us101.mat', t_h=t_h, enc_size =64, CAV_ratio=cav)
tsDataloader = DataLoader(tsSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn) # 

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()

lossVal = 0 # revised by Lei
count = 0

mae = 0
count_mae = 0

vehid = []
target_ID = []
target_Loc = []
pred_x = []
pred_y = []
T = []
dsID = []
#ts_cen = []
#ts_nbr = []
wt_a = []
#nbr_location = []
#print (len(tsDataloader.dataset) / 256)

num_test = 0

for i, data in enumerate(tsDataloader):
    #print (i)
    

    st_time = time.time()
    flag, hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds, targetID, targetLoc = data
    if flag == 0: # this happens when no target HDV in front
        continue
    num_test += hist.size()[1]
    #print (hist[:,33,:])
    vehid.append(veh_id) # CAV ID
    target_ID.append(targetID) # target HDV ID
    target_Loc.append(targetLoc) # target HDV location
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



    fut_pred,  weight_a = net(hist, nbrs, mask, lat_enc, lon_enc)

    #print (fut_pred.shape)
    #print (fut_pred[:, 33, 0])
    #print (fut_pred[:,33, 1])

    #print (fut[:, 33, 0])
    #print (fut[:, 33, 1])

    l, c = maskedMSETest(fut_pred, fut, op_mask)
    mae_l, mae_c = maskedMAETest(fut_pred, fut, op_mask)

    #print (out[:, 0])
    #out = out.detach().cpu().numpy()
    #print (type(out))
    #out = pd.DataFrame(out)
    #out.to_csv('out2.csv')
    
    fut_pred_x = fut_pred[:,:,0].detach()
    fut_pred_x = fut_pred_x.cpu().numpy()
    #print (type(fut_pred_x)) # (25, 128)
    fut_pred_y = fut_pred[:,:,1].detach()
    fut_pred_y = fut_pred_y.cpu().numpy()
    pred_x.append(fut_pred_x)
    pred_y.append(fut_pred_y)

    #print (weight_ha.size())
    #ts_cen.append(weight_ts_center[:, :, 0].detach().cpu().numpy())
    #ts_nbr.append(weight_ts_nbr[:, :, 0].detach().cpu().numpy())
    wt_a.append(weight_a[:, :, 0].detach().cpu().numpy())
    #print (nbr_loc)
    #nbr_location.append(np.array(nbr_loc))
    
    #print (len(pred))
    lossVal +=l.detach() # revised by Lei
    count += c.detach()

    mae += mae_l.detach()
    count_mae += mae_c.detach()
    
    #print (lossVal)
    #print (count)

    #break

vehid = sum(vehid, [])
vehid = pd.DataFrame(vehid)

target_ID = sum(target_ID, [])
target_ID = pd.DataFrame(target_ID)

target_Loc = sum(target_Loc, [])
target_Loc = pd.DataFrame(target_Loc)

T = sum(T, [])
T = pd.DataFrame(T)

dsID = sum(dsID, [])
dsID = pd.DataFrame(dsID)

pred_x = np.concatenate( pred_x, axis=1 )
pred_x = pd.DataFrame(pred_x)

pred_y = np.concatenate( pred_y, axis=1 )
pred_y = pd.DataFrame(pred_y)

#print (ts_cen)
#ts_cen = np.concatenate( ts_cen, axis=0)
#ts_cen = pd.DataFrame(ts_cen)

#ts_nbr = np.concatenate( ts_nbr, axis=0)
#ts_nbr = pd.DataFrame(ts_nbr)

wt_a = np.concatenate(wt_a, axis=0)
wt_a = pd.DataFrame(wt_a)



print ('lossVal is:', lossVal)
print('total test sample number:', num_test)

if metric == 'nll':
    print(lossVal / count)
else:
    print('RMSE for each step is:', torch.pow(lossVal / count,0.5)*0.3048)   # Calculate RMSE and convert from feet to meters
    
    print ('MAE for each step is:', mae / count_mae)
    print ('Overall RMSE is:', torch.pow(sum(lossVal) / sum(count), 0.5)*0.3048)
    print ('Overall MAE is:', sum(mae) / sum(count_mae))
