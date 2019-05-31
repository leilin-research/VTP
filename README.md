# VTP
Vehicle Trajectory Prediction with Deep Learning Models

# conv_social_comments: 
code from the paper 'Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018' with comments added.

# HA-LSTM:
code of hieratical-attention LSTM for vehicle trajectory prediction (time-step level and vehicle-level attention mechanisms).

# Data:
NGSIM data - the same processing procedure from the convolutional social pooling paper is followed. The training/val/testing datasets can be dowloaed from [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).

# 5-step RMSE:

Conv_social: [0.1029, 0.2023, 0.3146, 0.4364, 0.5674] training time: 3.30 hours

HA-LSTM: [0.0995, 0.2002, 0.3130, 0.4348, 0.5615]

LSTM: [0.1012, 0.2093, 0.3384, 0.4830, 0.6406]

LSTM with attention: [0.0984, 0.1961, 0.3095, 0.4318, 0.5601]

HA-LSTM (seq-to-seq): [0.1067, 0.2135, 0.3339, 0.4617, 0.5940]

