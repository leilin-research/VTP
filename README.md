# VTP: Deep Learning for Vehicle Trajectory Prediction

## Data
The training/validation/test datasets extracted from [Next Generation Simulation (NGSIM) Vehicle Trajectories](https://www.opendatanetwork.com/dataset/data.transportation.gov/8ect-6jqj) can be downloaded [here](https://drive.google.com/open?id=1dFMpX8HeCradMaCh4h0bD60h8k3M65Fw).

## STA-LSTM: An LSTM model with spatial-temporal attention mechanisms
- STA-LSTM utilizes *T*-step historical trajectories of all vehicles within a 3×13 grid centered around the target vehicle to predict its future trajectories. The architecture of STA-LSTM is shown below:

<p align="center">
  <img src="images/sta-lstm.png" width="750" height="400" />
</p>

### STA-LSTM achieves comparable prediction performance against other state-of-the-art models

<table>
<tr>
<td rowspan=2><b>Models</b> <td colspan=5><b>RMSE per prediction time step</b>
<tr>
<td colspan=1><b>1st</b> <td colspan=1><b>2nd</b> <td colspan=1><b>3rd</b> <td colspan=1><b>4th</b> <td colspan=1><b>5th</b>
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

### Average temporal-level attention weights of the past six time steps

<p align="center">
  <img src="images/temporal-weights.png" width="300" height="300" />
</p>

### Spatial-level attention weight analysis

- Distributions of spatial-level attention weights by target vehicle class (excluding weights in the target vehicle's cell):

<p align="center">
  <img src="images/spatial-class.png" width="600" height="300" />
</p>

- Averaged spatial-level attention weights by the number of neighboring vehicles

<p align="center">
  <img src="images/density.png" width="700" height="300" />
</p>

- Maximum spatial-level attention weight frequency by target vehicle location

<p align="center">
  <img src="images/101-attention.png" width="700" height="300" />
</p>

- Maximum spatial-level attention weights regrading the lane-changing behaviors of the target vehicle 2858. The target vehicle 2858 executed two lane-changing maneuvers around the 996th and 1220th time steps, respectively. The vehicle relocated the maximum attention from the current lane to (Right, 1) and (Right, 2), as it was preparing to change to the right lane.

<p align="center">
  <img src="images/attention_single_veh.png" width="400" height="500" />
</p>


## Citation
You are more than welcome to cite our paper.
```
@article{lin2020stalstm,
  title={Vehicle Trajectory Prediction Using LSTMs withSpatial-Temporal Attention Mechanisms},
  author={Lin, Lei and Li, Weizi and Bi, Huikun and Qin, Lingqiao},
  journal={IEEE Intelligent Transportation Systems Magazine (ITSM)}
}

```
