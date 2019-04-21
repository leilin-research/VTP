from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F

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
        self.enc_lstm1 = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Encoder LSTM
        self.enc_lstm2 = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        
        self.tanh = nn.Tanh()
        
        self.pre4att = nn.Sequential(
            #nn.ReLU(True),
            nn.Linear(self.encoder_size, 1), # (batch_size, sequenc_len, hidden_size) * (hidden_size, 1)
        )

        '''
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
        '''
        self.dec_lstm = torch.nn.LSTM(self.encoder_size, self.decoder_size) # 112, 128

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def attention_1(self, lstm_out_weight, lstm_out):
       
        alpha = F.softmax(lstm_out_weight, 1) # (batch_size, lstm_cell_num, 1), calculate weight along the 1st dimension
        #print (alpha.size()) #(10, 75, 1)
        #lstm_outs = lstm_outs.permute(0, 2, 1) # before: (batch_size,lstm_cell_num, hidden_dim); after: (batch_size, #hidden_dim, lstm_cell_num)
        lstm_out = lstm_out.permute(0, 2, 1)#128, 64, 16
         
        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2) # new_hidden_state-(batch_size, hidden_dim)
        new_hidden_state = F.relu(new_hidden_state)
         
        #print (new_hidden_state.size()) # (10, 20)
        #print (alpha)

        return new_hidden_state, alpha#, soft_attn_weights_1#, soft_attn_weights_2

    def attention_bidirec(self, lstm_outs, lstm_out_sum, lstm_out_weight):
        alpha = F.softmax(lstm_out_weight, 1) # (batch_size, lstm_cell_num, 1), calculate weight along the 1st dimension
        #print (alpha.size()) #(10, 75, 1)
        #lstm_outs = lstm_outs.permute(0, 2, 1) # before: (batch_size,lstm_cell_num, hidden_dim); after: (batch_size, hidden_dim, lstm_cell_num)
        lstm_out_sum = lstm_out_sum.permute(0, 2, 1)
        
        
        new_hidden_state = torch.bmm(lstm_out_sum, alpha).squeeze(2) # new_hidden_state-(batch_size, hidden_dim)
        new_hidden_state = F.relu(new_hidden_state)
        #print (new_hidden_state.size()) # (10, 20)
        #print (alpha)

        return new_hidden_state, alpha#, soft_attn_weights_1#, soft_attn_weights_2
    
    ## Forward Pass
    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):
        #print (hist.size())16, 128, 64
        ## Forward pass hist:
        lstm_out,(hist_enc,_) = self.enc_lstm1(self.leaky_relu(self.ip_emb(hist)))
        #print (lstm_out.size()) # (16, 128, 64) (seq_len, batch_size, hidden_state_size)
        lstm_out = lstm_out.permute(1, 0, 2) # 128, 16, 64
        #print (lstm_out.size())
        lstm_weight = self.pre4att(self.tanh(lstm_out)) # 128, 16, 1
       # print (lstm_weight.size())
        new_hidden, soft_attn_weights = self.attention_1(lstm_weight, lstm_out) # new_hidden: 128, 64
        '''
        lstm_out_sum = lstm_out[:,:,0:10] + lstm_out[:,:,10:20]
        #print (lstm_out_sum.size())
        lstm_weight = self.pre4att(self.tanh(lstm_out_sum)) #lstm_out
        new_hidden, soft_attn_weights = self.attention_bidirec(lstm_out, lstm_out_sum, lstm_weight)
	'''
   
       # print (lstm_out.size())
        
        new_hidden = new_hidden.unsqueeze(2) # (128, 64, 1)
        #print (new_hidden.size())
        #print ('hist_enc size is ', hist_enc.size()) # (128, 64, 1)
        
        ## Forward pass nbrs
        #print ('neighbor size is ', nbrs.size()) # example (16, 991, 2), 16 (history 30/ downsample 2), 991-all the number of neighbors in the past 30 time steps, 2-x and y locations
        nbrs_out, (nbrs_enc,_) = self.enc_lstm1(self.leaky_relu(self.ip_emb(nbrs)))
        # apply attention mechanism to neighbors
        nbrs_out = nbrs_out.permute(1, 0, 2) # 991, 16, 64
        #print (nbrs_out.size())
        nbrs_lstm_weight = self.pre4att(self.tanh(nbrs_out)) # 128, 16, 1
       # print (lstm_weight.size())
        new_nbrs_hidden, soft_nbrs_attn_weights = self.attention_1(nbrs_lstm_weight, nbrs_out) # new_hidden: 128, 64
        nbrs_enc = new_nbrs_hidden
        # end: apply attention mechanism to neighbors
        
        '''
        # not apply attention to neighbors
        #print ('before', nbrs_enc.size()) # example (1, 991, 64), 16 will be the timesteps in LSTM; 1 is because the output is from the last LSTM cell; 991 number of neighbors, 2-x and y locations
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2]) # squeeze the dimision 0, now becomes (991, 64)
        #print ('after', nbrs_enc.size())
        '''
        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float() # mask size: (128, 3, 13, 64)
        #print ('masks size is ', masks.size())
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc) # masked_scatter_(mask, source), Copies elements from source into self tensor at positions where the mask is one. Copy nbrs_enc values where masks is 0
        #*********************
        # this masked_scatter_ function is really important, 
        # soc_enc and masks must have the same shape
        # copy nbrs_enc values sequentially to soc_enc where masks is 1
        soc_enc = soc_enc.permute(0,3,2,1) # soc_enc size: (128, 64, 13, 3)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1)
        # print (soc_enc.size()) 128, 64, 39

        # concatenate hidden states:
        new_hs = torch.cat((soc_enc, new_hidden), 2)
        #print (new_hs.size()) # 128, 64, 40
        new_hs_per = new_hs.permute(0, 2, 1)
        
        # second attention
        weight = self.pre4att(self.tanh(new_hs_per)) # 128, 16, 1
        #print (weight.size()) 128, 40, 1
        new_hidden_ha, soft_attn_weights_ha = self.attention_1(weight, new_hs_per) # new_hidden: 128, 64
        #print (new_hidden_ha.size()) 128, 64
        ## Concatenate encodings:
        enc = new_hidden_ha #new_hidden.view(new_hidden.shape[1],new_hidden.shape[2])#torch.cat((soc_enc,hist_enc),1) # (128, 112)
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
            return fut_pred, soft_attn_weights, soft_nbrs_attn_weights, soft_attn_weights_ha


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





