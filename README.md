# VTP
Vehicle Trajectory Prediction with Deep Learning Models


## STA-LSTM: An LSTM model with spatial-temporal attention mechanisms
STA-LSTM achieves comparable prediction performance against other state-of-the-art models (e.g., conv-LSTM [1], naive LSTM), and also explains the influence of historical trajectories and neighboring vehicles on the target vehicle.

<table>
<tr>
<td colspan=3>a  <td colspan=2>b
<tr>
<td colspan=1>col1 <td colspan=1>col2 <td colspan=1>col3<td colspan=1>col4 <td colspan=1>col5
</table>


| Models     | <td colspan=5>RMSE per prediction time step |
|  ----    | ---       |---       |---       |---       |---       |

| physics-based model |  0.1        |2       |3       |4       |5      |
| naive LSTM     |         |2       |3       |4       |5      |
| SA-LSTM     |         |2       |3       |4       |5      |
| CS-LSTM     |         |2       |3       |4       |5      |
| STA-LSTM     |         |2       |3       |4       |5      |

[1] Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018

# Data:
The training/val/testing datasets can be dowloaed from [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).


