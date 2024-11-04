# Trajectory Data
## Columns Description
1. **Dataset ID**
   - **Description:** Identifier for the dataset source.
   - **Values:** Integers from 1 to 6, corresponding to different datasets.
2. **Vehicle ID**
   - **Description:** Unique identifier for each vehicle within a dataset.
3. **Frame Number**
   - **Description:** Timestamp or frame index indicating the time of the measurement.
4. **Local X**
   - **Description:** Lateral position of the vehicle in meters.
5. **Local Y**
   - **Description:** Longitudinal position of the vehicle in meters.
6. **Lane ID**
   - **Description:** Lane number where the vehicle is currently traveling.
7. **Lateral Maneuver**
   - **Description:** Indicates the vehicle's lateral movement.
   - **Values:**
     - `1` - Keeping lane.
     - `2` - Changing lane to the left.
     - `3` - Changing lane to the right.
8. **Longitudinal Maneuver**
   - **Description:** Indicates the vehicle's longitudinal movement.
   - **Values:**
     - `1` - Maintaining speed.
     - `2` - Slowing down.
9. **Neighbor Vehicle IDs at Grid Locations (Columns 9-47)**
   - **Description:** Vehicle IDs of neighboring vehicles within the spatial grid around the subject vehicle.
   - **Grid Structure:**
     - **Left Lane Grids (Columns 9-21):**
       - Grids 1 to 13 for the left adjacent lane.
     - **Same Lane Grids (Columns 22-34):**
       - Grids 14 to 26 for the same lane.
     - **Right Lane Grids (Columns 35-47):**
       - Grids 27 to 39 for the right adjacent lane.
   - **Values:**
     - Vehicle IDs if a vehicle is present in the grid cell.
     - Empty or placeholder value if no vehicle is present.


## Additional Information
- **Spatial Grid Details:**
  - **Longitudinal Range:** Each grid cell covers **15 meters**, spanning from **-90 meters** (behind) to **+90 meters** (ahead) relative to the subject vehicle.

# Tracks Data

The `tracks` data is a collection containing the complete trajectories of each vehicle in the dataset. It provides detailed temporal information about each vehicle's movement over time, focusing on:

- **Frame Numbers**: The sequence of time frames during which the vehicle was observed.
- **Local X Positions**: The lateral positions of the vehicle at each frame (in meters).
- **Local Y Positions**: The longitudinal positions of the vehicle at each frame (in meters).

## Structure of the `tracks` Data

- The `tracks` data is organized as a cell array (or a dictionary) indexed by:

  - **Dataset ID**: An integer from 1 to 6, indicating which dataset the vehicle belongs to.
  - **Vehicle ID**: A unique identifier for each vehicle within a dataset.

- Each entry in `tracks` corresponds to a specific vehicle and contains a matrix with:

  - **Row 1**: Frame Numbers (timestamps).
  - **Row 2**: Local X positions.
  - **Row 3**: Local Y positions.

- The data is transposed such that **columns represent time steps**, and **rows represent variables**.

## Accessing the `tracks` Data

To access the trajectory of a specific vehicle:

```matlab
vehicle_track = tracks{dataset_id, vehicle_id};
