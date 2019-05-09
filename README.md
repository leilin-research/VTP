# VTP
Vehicle Trajectory Prediction with Deep Learning Models

# conv_social_comments: 
code from the paper 'Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018' with comments added.

# HA-LSTM:
code of hieratical-attention LSTM for vehicle trajectory prediction (time-step level and vehicle-level attention mechanisms).

# Data:
NGSIM data - the same processing procedure from the convolutional social pooling paper is followed. The training/val/testing datasets can be dowloaed from [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).

# 5-step RMSE:

Conv_social: tensor([0.1002, 0.1964, 0.3096, 0.4328, 0.5594], device='cuda:0')

HA-LSTM: tensor([0.1038, 0.2060, 0.3192, 0.4397, 0.5658], device='cuda:0')

HA-LSTM (seq-to-seq): tensor([0.1067, 0.2135, 0.3339, 0.4617, 0.5940], device='cuda:0')

