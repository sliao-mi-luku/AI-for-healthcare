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


##
