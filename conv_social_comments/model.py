from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation

class highwayNet(nn.Module):

    ## Initialization
    def __init__(self,args):
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False)
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.soc_conv_depth = args['soc_conv_depth']
        self.conv_3x1_depth = args['conv_3x1_depth']
        self.dyn_embedding_size = args['dyn_embedding_size'] # 32
        self.input_embedding_size = args['input_embedding_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1)) # input channel, output channe, kernel size(3, 1): tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size) # 80+32+3+2, 128
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size) # 112, 128

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)


    ## Forward Pass
    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):

        ## Forward pass hist:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2]))) # dyn_emb a feedforward network

        ## Forward pass nbrs
        print ('neighbor size is ', nbrs.size()) # example (16, 991, 2), 16 (history 30/ downsample 2), 991-all the number of neighbors in the past 30 time steps, 2-x and y locations
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        #print ('before', nbrs_enc.size()) # example (1, 991, 64), 16 will be the timesteps in LSTM; 1 is because the output is from the last LSTM cell; 991 number of neighbors, 2-x and y locations
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2]) # squeeze the dimision 0, now becomes (991, 64)
        #print ('after', nbrs_enc.size())

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float() # mask size: (128, 3, 13, 64)
        #print ('masks size is ', masks.size())
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc) # masked_scatter_(mask, source), Copies elements from source into self tensor at positions where the mask is one. Copy nbrs_enc values where masks is 0
        #*********************
        # this masked_scatter_ function is really important, 
        # soc_enc and masks must have the same shape
        # copy nbrs_enc values sequentially to soc_enc where masks is 1
        soc_enc = soc_enc.permute(0,3,2,1) # soc_enc size: (128, 64, 13, 3)

        #print ('soc_enc size is: ', soc_enc.size())
        # input size (N, C_in, H, W) of the Conv2d, So input channel is 64, H is 13, width is 3

        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        # print ('soc_enc size is: ', soc_enc.size()) # (128, 16, 5, 1)
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)
        # print ('soc_enc size is: ', soc_enc.size())  # (128, 80)

        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        ## Concatenate encodings:
        enc = torch.cat((soc_enc,hist_enc),1) # (128, 112)
        #print (enc.size())


        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc)) # calculate probability of lateral movement
            lon_pred = self.softmax(self.op_lon(enc)) # calculate probability of longtudinal movement

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)  #(128, 117) 80 + 32 + 3 + 2 = 117 # lat_enc historical lateral movement information 128 by 3; lon_enc historical longitunal movement information
                #print (enc.size())
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc) 
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp)) # get six possible trajectories
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc) 
            return fut_pred


    def decode(self,enc):
        #print (enc.size()) # (128, 117)
        enc = enc.repeat(self.out_length, 1, 1)
        #print (enc.size()) # (25, 128, 117), why 25? because we want to predict future 50 timesteps histories and the downsampling rate is 2
        h_dec, _ = self.dec_lstm(enc) # (25, 128, 128)
        h_dec = h_dec.permute(1, 0, 2) # (128, 25, 128)
        fut_pred = self.op(h_dec) # (128, 25, 5)
        fut_pred = fut_pred.permute(1, 0, 2) # (25, 128, 5) 25 timesteps, 128 batchsize, 5 prediction
        fut_pred = outputActivation(fut_pred)
        return fut_pred





