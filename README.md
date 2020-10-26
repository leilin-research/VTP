# VTP
Vehicle Trajectory Prediction with Deep Learning Models


## STA-LSTM: An LSTM model with spatial-temporal attention mechanisms
STA-LSTM achieves comparable prediction performance against other state-of-the-art models (e.g., conv-LSTM [1], naive LSTM), and also explains the influence of historical trajectories and neighboring vehicles on the target vehicle.

<table>
<tr>
<td colspan=1>Models <td colspan=5>RMSE per prediction time step
<tr>
<td colspan=1> <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5
<tr>
<td colspan=1>physics-based model <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5
<tr>
<td colspan=1>naive LSTM <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5
<tr>
<td colspan=1>SA-LSTM <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5
<tr>
<td colspan=1>CS-LSTM <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5
<tr>
<td colspan=1>STA-LSTM <td colspan=1>1 <td colspan=1>2<td colspan=1>3 <td colspan=1>4 <td colspan=1>5

</table>



[1] Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018

# Data:
The training/val/testing datasets can be dowloaed from [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).


