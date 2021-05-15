# Medical_image_analysis

last updated: 05/14/2021

Reference: Udacity's AI for Healthcare Nanodegree

---

## DICOM

DICOM, Digital Imaging and Communication in Medicine, is a standard of how medical imaging data are **stored** and **transmitted over networks**.

An medical imaging dataset on a patient consists of multiple **studies**, and each study consists of multiple **series**, where each each series is made of multiple images.

Example: a single MRI series consist of several 2D images scanned in a single acquisition sweep.

A DICOM data contains many tags (a dim=2 tuple), whcih uniquely describes a piece of information assiciated with the data

You can look up the tags in the link: [Registry of DICOM data elements](http://dicom.nema.org/medical/dicom/2020a/output/chtml/part06/chapter_6.html)

You can lool up the definition of each data type in this link: [DICOM value representation](http://dicom.nema.org/medical/dicom/2020a/output/chtml/part05/sect_6.2.html)
 
**Why important?**
  
- DICOM can be seen as a universal way that medical imaging dataset are formatted
- Therefore, getting used to the **anatomy** and **data format** of DICOM makes it easier to\
  -- dive into the data quicker\
  -- know (or know how to find) the location where an information you want to retrieve is located (tag)\
  -- know (or know how to find) the way the information is encoded/represented (value representation)

**Hands-on tips**

For example, if you want to know the age of the patient in the data:

1. You just type in "DICOM age" in Google and you can find that the tag is (0010,1010), and the **Value representation (VR)** is Age String (AS)
2. You then go to [this page](http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html), and you can see the definition of AS values
3. Given the definition written on the page, it seems that if the patient is 30 years old, the value will be 030Y

Softwares you can use to view DICOM images: **MicroDicom**



---
## NIFTI format

NIFTI (Neuroimaging Informatics Technology Initiative) is a standard to store neurological imaging data.

A few things to know:

1. The +x axis of NIFTI is pointing to the right side of the patient. The +x axis of DICOM is point to the left side of the patient
2. NIFTI can store multiple images in a single file
3. NIFTI does not store as many meta data as DICOM
4. NIFTI may has it's own-defined units of measurement

**Why important?**

- Many medical images in the wild are in NIFTI format (especially in ML/DS competitions)
- If you are familiar with NIFTI format, you can easiy write a piece of code to load data and assemble your ML pipeline
- And you can move on to EDA quickly!


**Hands-on tips**

Softwares you can use to view NIFTI images: **3D Slicer**

The python *pydicom* is the tool yo want to use to handle DICOM images



---
## Exploratory Data Analysis (Python)

**Loading images**

```python3
import pydicom
import numpy as np
import os

path = f"<path_to_the_dataset>"

# loading images
slices = [pydicom.dcmread(os.path.join(path, f)) for f in sorted(os.listdir(path))]

# sort the slices by the correct order
slices = sorted(slices, key = lambda x: x.ImagePositionPatient[0])
```

**Looking up metadata information**

```python3

```
