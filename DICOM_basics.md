# DICOM Basics

Resources:

1. [DICOM standard](https://www.dicomstandard.org/)
2. [DICOM SOPs](http://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html)
3. [DICOM data elements](http://dicom.nema.org/medical/dicom/current/output/chtml/part06/chapter_6.html)
4. [DICOM value representation](http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html)


## DICOM

DICOM, Digital Imaging and Communications in Medicine, is a standard:

1. Defines a way to store medical images
2. Defines how medical images data is transmitted over networks

**Hierarchy**

- Patient
- Study: a medical study performed on a patient with a scanner
- Series: a single acquisition sweep
- Instance: a single scan image (pixel data + metadata)


## Viewing tools

1. MicroDicom

2. 3D Slicer

## Important parameters

(0020, 0037) Image Orientation Patient - define the orientation of the first row and the first column of the image.

(0020, 0032) Image Position Patient - define the (x, y, z) coordinates of the upper left corner of the image

(0028, 0030) Pixel Spacing - physical distance between pixel centers

(0018, 0050) Slice Thickness - thickness of a single slice

(0028, 0100) Bitts Allocated - number of bits allocated for each pixel

(0028, 0101) Bits Stored - number of bits actually used by each pixel

(0020, 0010) Rows - height of the slice (in voxels)

(0020, 0011) Columns - width of the slice (in voxels)

```python
"""
Example codes
"""

import numpy as np
import pydicom
import os

dcm_dir_path = 'the/path/to/the/dcm/file'

# read all dcm files in the dcm_dir_path and append them to a list (slices)
slices = [pydicom.dcmread(os.path.join(dcm_dir_path, f)) for f in os.listdir(dcm_dir_path)]

# print all stored DICOM metadata
print(slices[0])

# view pixel spacing
print("Pixel Spacing = {}".format(slices[0].PixelSpacing))
# or
print("Pixel Spacing = {}".format(slices[0][0x0028, 0x0030].value))

# retrieve pixel data
image_data = np.stack([s.pixel_array for s in slices])
print(image_data.shape)
print(image_data.dtype)

# display a slice
img = image_data[100, :, :]
plt.imshow(img, cmap="gray")

# adjust the scale
aspect_ratio = slices[0].SliceThickness / slices[0].PixelSpacing[0]
plt.imshow(img, cmap="gray", aspect=aspect_ratio)

```


## Houndfield Units (CT scans)

| Type | Hounsfield Units |
| ----------- | ----------- |
| Bone | 400 ~ 1,000 |
| Soft Tissue | 80 ~ 400 |
| Water | 0 |
| Fat | -100 ~ -60 |
| Lung | -600 ~ -400 |
| Air | -1,000 |


## Loading NIFTI volumes

``` python
import nibabel as nib

nii_img = nib.load('path/to/the/nii.gz/file')

img = nii_img.get_fdata()

# pixel dimension
print(nii_img.header["pixdim"])
```

## Segmentation

```python

```

**Metrics**

Sensitivity = TP / (TP + FN)

> Low Sensitivity --> under segmentation

Specificity = TP / (TP + FP)

> Low Specificity --> over segmentation

Dice Similarity Coefficient (DSC) = 2|X and Y| / (|X| + |Y|)

Jaccard Index (J) = |X and Y| / |X or Y|

Hausdorff Distance HD = max(Dist_X_to_Y, Dist_Y_to_X)

## DICOM Networking

Definition resources: http://dicom.nema.org/medical/dicom/2020a/output/chtml/part07/PS3.7.html

#### Application Entity

An abstraction of an application that uses DICOM networking protocol to communicate

#### DIMSE (DICOM Message Service Element)

#### PACS (Picture Archiving and Communication System)

#### VNA (Vendor Neutral Archive)

Often deployed in a cloud environment

#### EHR (Electronic Health Record)

#### RIS (Radiology Information System)

Mini-EHRs for radiology departments

#### HL7 (Health Level 7)

A protocol used to exchange patient data and physician orders between systems

#### FHIR (Fast Healthcare Interoperability Resources)

The new generation of HL7






















ddd
