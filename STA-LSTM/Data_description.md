
## Columns Description

After processing, the dataset contains the following columns:

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

- **Data Frequency:**
  - The dataset records vehicle positions at **10 frames per second**.
- **Spatial Grid Details:**
  - **Longitudinal Range:** Each grid cell covers **15 meters**, spanning from **-90 meters** (behind) to **+90 meters** (ahead) relative to the subject vehicle.

