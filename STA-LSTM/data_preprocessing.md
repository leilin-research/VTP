# NGSIM Dataset Processing Steps and Columns Description

## Dataset Processing Steps

### **1. Data Loading and Preprocessing**

- **Load Raw Data:**
  - Load 6 raw NGSIM files for:
    - US-101 freeway
    - I-80 freeway
- **Assign Dataset IDs:**
  - Assign a unique identifier (from 1 to 6) to each dataset to distinguish between different locations and time periods.
- **Select Relevant Fields:**
  - Extract the following columns for further analysis:
    1. **Dataset ID**
    2. **Vehicle ID**
    3. **Frame Number**
    4. **Local X** (lateral position)
    5. **Local Y** (longitudinal position)
    6. **Lane ID**
- **Standardize Lane IDs:**
  - For datasets from the US-101 freeway (datasets 1-3), cap the maximum lane ID at 6 to ensure consistency across datasets.

### **2. Maneuver Calculation**

#### **Lateral Maneuvers (Lane Changes)**

- **Objective:** Determine if a vehicle is keeping its lane, changing lanes to the left, or changing lanes to the right.
- **Method:**
  - For each vehicle at each time frame:
    - **Look-Ahead and Look-Back Windows:**
      - **Look-Ahead Window:** 40 frames into the future (~4 seconds).
      - **Look-Back Window:** 40 frames into the past (~4 seconds).
    - **Compare Lane Positions:**
      - Compare the current lane ID with the lane IDs at the upper and lower bounds of the window.
    - **Assign Maneuver Codes:**
      - `1` - Keeping lane.
      - `2` - Changing lane to the left.
      - `3` - Changing lane to the right.
- **Criteria:**
  - If future lane ID > current lane ID or current lane ID > past lane ID, it's a **right lane change**.
  - If future lane ID < current lane ID or current lane ID < past lane ID, it's a **left lane change**.
  - Otherwise, the vehicle is **keeping its lane**.

#### **Longitudinal Maneuvers (Speed Changes)**

- **Objective:** Identify whether a vehicle is maintaining its speed or slowing down significantly.
- **Method:**
  - For each vehicle at each time frame:
    - **Define Time Windows:**
      - **Past Window:** 30 frames before the current frame (~3 seconds).
      - **Future Window:** 50 frames after the current frame (~5 seconds).
    - **Calculate Speeds:**
      - **Historical Speed (`vHist`):** Average speed over the past window.
      - **Future Speed (`vFut`):** Average speed over the future window.
    - **Compare Speeds:**
      - Calculate the ratio `vFut / vHist`.
    - **Assign Maneuver Codes:**
      - `1` - Maintaining speed.
      - `2` - Slowing down (if `vFut / vHist < 0.8`).

### **3. Neighbor Identification**

- **Objective:** Map the positions of surrounding vehicles relative to each vehicle at each time frame.
- **Method:**
  - **Define a Spatial Grid:**
    - Longitudinal range: **-90 meters** to **+90 meters** relative to the vehicle.
    - Grid cell size: **15 meters**.
    - Total grids per lane: **13** (for the longitudinal axis).
    - Lanes considered:
      - Left adjacent lane.
      - Same lane.
      - Right adjacent lane.
  - **Assign Neighbor Vehicles to Grid Cells:**
    - For each vehicle, identify neighboring vehicles within the spatial grid.
    - Calculate the relative longitudinal distance (`ΔY`) between the neighboring vehicle and the subject vehicle.
    - Assign the neighbor's vehicle ID to the corresponding grid cell based on `ΔY`.
  - **Grid Indexing:**
    - **Left Lane Grids (Grids 1-13):** Indices 1 to 13.
    - **Same Lane Grids (Grids 14-26):** Indices 14 to 26.
    - **Right Lane Grids (Grids 27-39):** Indices 27 to 39.

### **4. Data Splitting**

- **Objective:** Divide the dataset into training, validation, and test sets without overlap in vehicle IDs.
- **Method:**
  - **Combine All Datasets:**
    - Merge all processed trajectory data into a single dataset.
  - **Split by Vehicle IDs:**
    - For each dataset (1 to 6), sort the vehicle IDs.
    - Determine split thresholds:
      - **Training Set:** Vehicles with IDs in the first 70%.
      - **Validation Set:** Vehicles with IDs in the next 10%.
      - **Test Set:** Vehicles with IDs in the last 20%.
  - **Ensure Uniqueness:**
    - Vehicles in the training set are not present in the validation or test sets, and so on.

### **5. Edge Case Filtering**

- **Objective:** Remove data points where vehicles do not have sufficient trajectory history or future data.
- **Method:**
  - **Check Trajectory Length:**
    - For each data point, verify that the vehicle's trajectory includes at least **3 seconds** (30 frames) of past data.
    - Ensure there is future data available beyond the current frame.
  - **Filter Data Points:**
    - Retain only those data points that meet the criteria.
    - This ensures the model has enough historical data for prediction and learning.

### **6. Data Saving**

- **Objective:** Save the processed datasets for future use.
- **Method:**
  - **Save to `.mat` Files:**
    - For each of the training, validation, and test sets:
      - Save the trajectory data (`traj`) and the vehicle tracks (`tracks`) into separate `.mat` files:
        - `TrainSet.mat`
        - `ValSet.mat`
        - `TestSet.mat`
  - **Contents of Saved Files:**
    - **`traj`:** Contains the processed trajectory data with maneuvers and neighbor information.
    - **`tracks`:** Contains the full trajectory (time, position) of each vehicle.
