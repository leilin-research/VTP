from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import h5py
import hdf5storage
import random
import os

#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):

    def __seed_torch(seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def __init__(self, mat_file, t_h=30, t_f=40, d_s=2, enc_size = 64, grid_size = (265,3), CAV_ratio = 0.7):
        i = 0
        print (mat_file)
        self.D = hdf5storage.loadmat(mat_file)['res_traj'] 
        print (self.D.shape)
        self.T = hdf5storage.loadmat(mat_file)['res_t'] 
        print (self.T.shape)
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid
        self.cav_ratio = CAV_ratio # ratio of CAVs



    def __len__(self):
        return len(self.D)



    def __getitem__(self, idx):

        #self.__seed_torch()
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,10:] # sarting from 10 not 8 anymore
        neighbors = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId,t,vehId,dsId) # history of the same target vehicle
        fut = self.getFuture(vehId,t,dsId) # future of the target vehicle

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        indx = 0 # index of the vehicle in left, current, right lanes
        f = 1 # control how far a CV can communicate in the front

        for i in grid: # neighbors found from data preprocessing
            #if (indx >= 132 and indx <= f*264) or (indx >= 397 and indx <= f*529) or (indx >= 662 and indx <= f*794): # only pick CAVs that are in the front of the target vehicle
            if random.uniform(0, 1) <= self.cav_ratio: # assume this neighbor is CAV
            	neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId)) # history of neighbor vehicles
            else: # assum this neighbor is not a CAV
                neighbors.append(np.empty([0, 2]))  
            indx = indx + 1 # index of the vehicle in left, current, right lanes
        
        indx = 0 # reloop the neighbors, find the target HDV
        flag = 0 # indicate whether to keep this record or not
        final_targetID = 0
        location_targetID = 0
        for i in grid:
            if indx == 397: # the location of the ego CAV the first row has 265 grid cells, the second row to the ego CAV has 133 rows, total: 265+133-1
                neighbors[indx] = self.getHistory(vehId,t,vehId,dsId) # the ego CAV is a neighbor for the target HDV
            if indx > 397 and indx <= 529:#403: # find the target HDV in the front grids, which forms the scenario that an ego CAV follows a target HDV
                tem = self.getHistory(i.astype(int), t,vehId,dsId)
                if len(tem) != 0: 
                    targetHDVId = i.astype(int)
                    egoCAVId = vehId
                    hist = self.getHistory(targetHDVId,t,vehId,dsId) # the historical trajectory of the target HDV
                    fut = self.getFuture_targetHDV(targetHDVId, egoCAVId, t,dsId)# replace the future trajectory of the center vehicle with that of the target HDV
                    #print (fut)
                    neighbors[indx] = tem # insert the target HDV into the neighbor trajectories
                    flag = 1 # useful record, keep it!!!!!!!!!
                    #tem_traj.append(tem)
                    #tem_id.append(indx)
            if flag == 1: # previously forgot to add this, we want to find the nearest one
                final_targetID = targetHDVId
                location_targetID = indx
                break
            indx = indx + 1 

        lon_enc = np.zeros([2])
        lat_enc = np.zeros([3])


        return flag, hist, fut, neighbors, lat_enc, lon_enc, vehId, t, dsId, final_targetID, location_targetID



    ## Helper function to get track history
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if self.T.shape[1]<=vehId-1:
                return np.empty([0,2])
            refTrack = self.T[dsId-1][refVehId-1].transpose() # take the target vehicle as the reference point
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3] # 1:3: time and x, y

            if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos # start, end, step is the downsample (self.d_s)

            if len(hist) < self.t_h//self.d_s + 1:
                return np.empty([0,2])
            return hist



    ## Helper function to get track future
    def getFuture(self, vehId, t,dsId):
        #print (dsId)
        #print (vehId)
        #print (self.T.shape)
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut

    ## Helper function to get track future
    def getFuture_targetHDV(self, targetHDVId, egoCAVId, t,dsId):
        #print (dsId)
        #print (vehId)
        #print (self.T.shape)
        HDVTrack = self.T[dsId-1][targetHDVId-1].transpose()
        CAVTrack = self.T[dsId-1][egoCAVId-1].transpose()
        refPos = CAVTrack[np.where(CAVTrack[:, 0] == t)][0, 1:3]

        stpt = np.argwhere(HDVTrack[:, 0] == t).item() + self.d_s # "+ self.d_s" means for the next t+1 step
        enpt = np.minimum(len(HDVTrack), np.argwhere(HDVTrack[:, 0] == t).item() + self.t_f + 1)
        fut = HDVTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut



    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        useful_samples = 0
        for flag,_,_,nbrs,_,_,_,_,_,_,_ in samples:
            if flag == 1:
                nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))]) # pick neighbors that are not zeros
                useful_samples += 1

        #print ('useful_samples is', useful_samples)
        maxlen = self.t_h//self.d_s + 1
        if nbr_batch_size == 0: # this happens when CAV ratio is very low, no CAVs in front
            nbr_batch_size = 20 # just assign a number, because the features will be 0, so it wont make a diff, this is just to prevent the fail when there are no neighbors
        nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)


        # Initialize social mask batch:
        pos = [0, 0]
        if useful_samples == 0:
            useful_samples = 20 # like above, assign a number to prevent warning
        mask_batch = torch.zeros(useful_samples, self.grid_size[1],self.grid_size[0],self.enc_size) # gird_size (13, 3) width 3, height 13, depth lenth of output of lstm cells
        mask_batch = mask_batch.byte() # self.to(torch.uint8)


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen,useful_samples,2) # len(samples)
        fut_batch = torch.zeros(self.t_f//self.d_s,useful_samples,2) # len(samples)
        op_mask_batch = torch.zeros(self.t_f//self.d_s,useful_samples,2) # future timestep, number of vehs, two locations x and y  # len(samples)
        lat_enc_batch = torch.zeros(useful_samples,3)  # len(samples)
        lon_enc_batch = torch.zeros(useful_samples, 2) # len(samples)


        count = 0
        veh_ID = []
        time = []
        dsID = []
        targetID = []
        targetLoc = []
        i = 0
        for sampleId,(flag, hist, fut, nbrs, lat_enc, lon_enc, vehId, t, ds,final_targetID, location_targetID) in enumerate(samples):
            if flag == 0:
                continue
            #print ('sampleId is: ', sampleId)
	    # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), i,0] = torch.from_numpy(hist[:, 0]) # NOTE here it is [0:len(hist), ...] not maxlen
            hist_batch[0:len(hist), i, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), i, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), i, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), i,:] = 1 # SIMILAR, here it is len(fut), not self.t_f//self.d_s
            lat_enc_batch[i,:] = torch.from_numpy(lat_enc)
            lon_enc_batch[i, :] = torch.from_numpy(lon_enc)
            veh_ID.append(vehId)
            time.append(t)
            dsID.append(ds)
            targetID.append(final_targetID)
            targetLoc.append(location_targetID)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0: # recall that in getHistory(), if there is no neighbor, it returns an empty array [0, 2], the length of which is 0
                    #print ('count is: ', count)
                    nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
		    # id is from 0-38, because each nbrs is a list of length 39, if there is neighbor, there is values, otherwist it is appended [0, 0]
                    pos[0] = id % self.grid_size[0] # qu yu
                    pos[1] = id // self.grid_size[0] # qu zheng, self.grid_size[0] is 13; < 13, pos[1] = 0, left; =13, pos[1] = 1, middle;
                    #print ('id is ', id)
                    #print ('self.grid_size[0] is ', self.grid_size[0])
                    #print ('pos[0] is ', pos[0])
                    #print ('pos[1] is ', pos[1])
                    mask_batch[i,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                    count+=1 # is it useful here?
            i += 1

        # i can indicate whether this batch is useful
        return i, hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, veh_ID, time, dsID, targetID, targetLoc


#________________________________________________________________________________________________________________________________________





## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2)
    return out

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    #print (muX)
    #print (muY)
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out #why x and y loss are both "out"?
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask) # although both uses out, the average will be correct, 2*out/2
    return lossVal

## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out #why x and y loss are both "out"?
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

'''
## MAPE loss for complete sequence, outputs a sequence of MAPE values, uses mask for variable output lengths, used for evaluation
def maskedMAPETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)), 0.5) / torch.pow((torch.pow(x, 2) + torch.pow(y, 2)), 0.5)
    out_noninf = out
    out_noninf[out == float("Inf")] = 0
    mask[out == float("Inf")] = 0
    
    #print (out)
    acc[:, :, 0] = out_noninf #why x and y loss are both "out"?
    acc[:, :, 1] = out_noninf
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts
'''
## MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMAETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow((torch.pow(x - muX, 2) + torch.pow(y - muY, 2)), 0.5)
    acc[:, :, 0] = out #why x and y loss are both "out"?
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
