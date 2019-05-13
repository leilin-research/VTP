# VTP
Vehicle Trajectory Prediction with Deep Learning Models

# conv_social_comments: 
code from the paper 'Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018' with comments added.

# HA-LSTM:
code of hieratical-attention LSTM for vehicle trajectory prediction (time-step level and vehicle-level attention mechanisms).

# Data:
NGSIM data - the same processing procedure from the convolutional social pooling paper is followed. The training/val/testing datasets can be dowloaed from [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).

# 5-step MSE:

Conv_social: [0.1011, 0.1976, 0.3109, 0.4324, 0.5599] training time: 3.30 hours

HA-LSTM: [0.1007, 0.1994, 0.3122, 0.4347, 0.5631]

HA-LSTM (seq-to-seq): [0.1067, 0.2135, 0.3339, 0.4617, 0.5940]

