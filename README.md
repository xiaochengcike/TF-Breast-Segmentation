# TF-Breast-Segmentation
Convolutional Neural Network for Breast Tumor Segmentation

## Dataset

The MRI scans and ground truth are layed out in the following structure:
```
├── Data/
│   ├── GroundTruth/
│   │   ├── 1001.nii
│   │   ├── ...
│   ├── Scans/
│   │   ├── 1001.nii
|   |   ├── ...
```

## TODO
- implement data augmentation
- implement U-Net itself
- fix data parser to create a uniform data shape
- integrate parser and U-Net
