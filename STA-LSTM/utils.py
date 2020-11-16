from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch

#___________________________________________________________________________________________________________________________

### Dataset class for the NGSIM dataset
class ngsimDataset(Dataset):


    def __init__(self, mat_file, t_h=30, t_f=10, d_s=2, enc_size = 64, grid_size = (13,3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences
        self.enc_size = enc_size # size of encoder LSTM
        self.grid_size = grid_size # size of social context grid



    def __len__(self):
        return len(self.D)



    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:]
        neighbors = []

        hist = self.getHistory(vehId,t,vehId,dsId) # history of the same target vehicle
        fut = self.getFuture(vehId,t,dsId) # future of the target vehicle

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid: # neighbors found from data preprocessing
            neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId)) # history of neighbor vehicles

        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 7] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 6] - 1)] = 1


        return hist,fut,neighbors,lat_enc,lon_enc, vehId, t, dsId



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
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
        return fut



    ## Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _,_,nbrs,_,_,_,_,_ in samples:
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))]) # pick neighbors that are not zeros
        maxlen = self.t_h//self.d_s + 1

        if nbr_batch_size != 0:
            nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)

        
            # Initialize social mask batch:
            pos = [0, 0]
            mask_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size) # gird_size (13, 3) width 3, height 13, depth lenth of output of lstm cells
            mask_batch = mask_batch.byte() # self.to(torch.uint8)


            # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
            hist_batch = torch.zeros(maxlen,len(samples),2)
            fut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2)
            op_mask_batch = torch.zeros(self.t_f//self.d_s,len(samples),2) # future timestep, number of vehs, two locations x and y
            lat_enc_batch = torch.zeros(len(samples),3)
            lon_enc_batch = torch.zeros(len(samples), 2)


            count = 0
            veh_ID = []
            time = []
            dsID = []
            for sampleId,(hist, fut, nbrs, lat_enc, lon_enc, vehId, t, ds) in enumerate(samples):

	    # Set up history, future, lateral maneuver and longitudinal maneuver batches:
                hist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0]) # NOTE here it is [0:len(hist), ...] not maxlen
                hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
                fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
                fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
                op_mask_batch[0:len(fut),sampleId,:] = 1 # SIMILAR, here it is len(fut), not self.t_f//self.d_s
                lat_enc_batch[sampleId,:] = torch.from_numpy(lat_enc)
                lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
                veh_ID.append(vehId)
                time.append(t)
                dsID.append(ds)

            # Set up neighbor, neighbor sequence length, and mask batches:
                for id,nbr in enumerate(nbrs):
                    if len(nbr)!=0: # recall that in getHistory(), if there is no neighbor, it returns an empty array [0, 2], the length of which is 0

                        nbrs_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0])
                        nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
		    # id is from 0-38, because each nbrs is a list of length 39, if there is neighbor, there is values, otherwist it is appended [0, 0]
                        pos[0] = id % self.grid_size[0] 
                        pos[1] = id // self.grid_size[0] 

                        mask_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte()
                        count+=1 

            return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, veh_ID, time, dsID #output mask
        else:
            return [-1], -1, -1, -1, -1, -1, -1, -1, -1, -1

#________________________________________________________________________________________________________________________________________





## Custom activation for output layer (Graves, 2015)
def outputActivation(x):
    muX = x[:,:,0:1]
    muY = x[:,:,1:2]

    out = torch.cat([muX, muY],dim=2)
    return out

## Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out 
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
    acc[:, :, 0] = out 
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts

