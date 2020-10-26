# VTP
Vehicle Trajectory Prediction with Deep Learning Models


## STA-LSTM: An LSTM model with spatial-temporal attention mechanisms
STA-LSTM achieves comparable prediction performance against other state-of-the-art models (e.g., conv-LSTM [1], naive LSTM), and also explains the influence of historical trajectories and neighboring vehicles on the target vehicle.


| Models      |          RMSE per prediction time step           ||
              | 1 | 2 | 3 | 4 | 5 | 
 ------------ | :-----------: | -----------: |:-----------: | -----------: |:-----------: |
physics-based model|  0.1776     |   0.3852     |    0.6033    |    0.8377    |    1.0888    |
naive LSTM |       |        |        |        |        |
SA-LSTM   |       |        |        |        |        |
CS-LSTM [1] |       |        |        |        |        |
STA-LSTM |      |        |        |        |        || 


<table>
<tr>
<td colspan=1>Models <td colspan=5>RMSE per prediction time step
<tr>
<td colspan=1> <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5
<tr>
<td colspan=1>physics-based model <td colspan=1>0.1776 <td colspan=1>0.3852 <td colspan=1>0.6033 <td colspan=1>0.8377 <td colspan=1>1.0888
<tr>
<td colspan=1>naive LSTM <td colspan=1>0.1012 <td colspan=1>0.2093 <td colspan=1>0.3384 <td colspan=1>0.4830 <td colspan=1>0.6406
<tr>
<td colspan=1>SA-LSTM <td colspan=1>0.1026 <td colspan=1>0.2031 <td colspan=1>0.3157 <td colspan=1>0.4367 <td colspan=1>0.5643
<tr>
<td colspan=1>CS-LSTM [1] <td colspan=1>0.1029 <td colspan=1>0.2023 <td colspan=1>0.3146 <td colspan=1>0.4364 <td colspan=1>0.5674
<tr>
<td colspan=1>STA-LSTM <td colspan=1>0.0995 <td colspan=1>0.2002 <td colspan=1>0.3130 <td colspan=1>0.4348 <td colspan=1>0.5615

</table>



[1] Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018

# Data:
The training/val/testing datasets can be dowloaed from [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).


