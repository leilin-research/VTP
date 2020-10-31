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

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.input_embedding_size = args['input_embedding_size']


        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm1 = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Encoder LSTM
        self.enc_lstm2 = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)
        
        self.spatial_embedding = nn.Linear(5, self.encoder_size)

        self.tanh = nn.Tanh()
        
        self.pre4att = nn.Sequential(
            #nn.ReLU(True),
            nn.Linear(self.encoder_size, 1), # (batch_size, sequenc_len, hidden_size) * (hidden_size, 1)
        )

        self.dec_lstm = torch.nn.LSTM(self.encoder_size, self.decoder_size) # 64, 128

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def attention(self, lstm_out_weight, lstm_out):
       
        alpha = F.softmax(lstm_out_weight, 1) # (batch_size, lstm_cell_num, 1), calculate weight along the 1st dimension

        lstm_out = lstm_out.permute(0, 2, 1)
         
        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2) # new_hidden_state-(batch_size, hidden_dim)
        new_hidden_state = F.relu(new_hidden_state)
         

        return new_hidden_state, alpha#, soft_attn_weights_1#, soft_attn_weights_2
    
    ## Forward Pass
    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):
        #print (hist.size())#16, 128, 2
        ## Forward pass hist:

        lstm_out,(hist_enc,_) = self.enc_lstm1(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = hist_enc.squeeze()
        hist_enc = hist_enc.unsqueeze(2)

        nbrs_out, (nbrs_enc,_) = self.enc_lstm2(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2]) # squeeze the dimision 0, now becomes (991, 64)

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float() 
        #print ('masks size is ', masks.size())
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc) # masked_scatter_(mask, source), Copies elements from source into self tensor at positions where the mask is one. Copy nbrs_enc values where masks is 0
        #*********************
        # this masked_scatter_ function is really important, 
        # soc_enc and masks must have the same shape
        # copy nbrs_enc values sequentially to soc_enc where masks is 1
        masks_tem = masks.permute(0, 3, 2, 1)
        #print (masks_tem.size()[0])
        soc_enc = soc_enc.permute(0,3,2,1) 
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1) 

        new_hs = torch.cat((soc_enc, hist_enc), 2)


        new_hs_per = new_hs.permute(0, 2, 1) 
        
        # second attention
        weight = self.pre4att(self.tanh(new_hs_per)) 

        new_hidden_ha, soft_attn_weights_ha = self.attention(weight, new_hs_per) 

        ## Concatenate encodings:
        enc = new_hidden_ha 

    

        fut_pred = self.decode(enc) 
        
        return fut_pred, soft_attn_weights_ha

# soft_attn_weights and soft_nbrs_attn_weights are the attention weights across time steps (the ego-vehicle and neighbors)
# soft_attn_weights_ha are the attention weights across vehicles (13 by 3 neighbors, row-wise flattern, [[1, 2, 3], [4, 5, 6], ...[X, X, X]], the 40th is the ego vehicle)


    def decode(self,enc):

        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc) 
        h_dec = h_dec.permute(1, 0, 2) 
        fut_pred = self.op(h_dec) 
        fut_pred = fut_pred.permute(1, 0, 2) 
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    def decode_by_step(self,enc):

        pre_traj = []

        decoder_input = enc

        for _ in range(self.out_length):
            decoder_input = decoder_input.unsqueeze(0)
            h_dec, _ = self.dec_lstm(decoder_input) 
            h_for_pred = h_dec.squeeze()
            fut_pred = self.op(h_for_pred) 
            pre_traj.append(fut_pred.view(fut_pred.size()[0], -1))
            
            embedding_input = fut_pred
            decoder_input = self.spatial_embedding(embedding_input)

        pre_traj = torch.stack(pre_traj, dim=0)
        pre_traj = outputActivation(pre_traj)
        return pre_traj







