# Homography-Estimation
3DCV

## homography estimation (DLT and NDLT) - 1.py
### Input
two images and their corresponding points

### Direct Linear Transform
1. 1-0 vs 1-1
    ```bash
    python 1.py images/1-0.png images/1-1.png groundtruth_correspondences/correspondence_01.npy DLT
    ```
2. 1-0 vs 1-2
    ```bash
    python 1.py images/1-0.png images/1-2.png groundtruth_correspondences/correspondence_02.npy DLT
    ```

### Normalized Direct Linear Transform
1. 1-0 vs 1-1
    ```bash
    python 1.py images/1-0.png images/1-1.png groundtruth_correspondences/correspondence_01.npy NDLT
    ```
2. 1-0 vs 1-2
    ```bash
    python 1.py images/1-0.png images/1-2.png groundtruth_correspondences/correspondence_02.npy NDLT
    ```

### Output
- Homography matrix
- Reprojection error


## Document rectification - 2.py
### Input
one image to be rectified

### Select 4 corner points
Using mouse_click_example.py to select 4 corner points
```bash
python mouse_click_example.py images/2-my_img.png
```

### Homography estimation and Warping
```bash
python 2.py images/2-my_img.png
```
### Output
rectified image 