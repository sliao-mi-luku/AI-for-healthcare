# Medical_image_analysis

last updated: 04/30/2021

---

## DICOM

DICOM, Digital Imaging and Communication in Medicine, is a standard of how medical imaging data are **stored** and **transmitted over networks**.

An medical imaging dataset on a patient consists of multiple **studies**, and each study consists of multiple **series**, where each each series is made of multiple images.

Example: a single MRI series consist of several 2D images scanned in a single acquisition sweep.

A DICOM data contains many tags (a dim=2 tuple), whcih uniquely describes a piece of information assiciated with the data

You can look up the tags in the link: [Registry of DICOM data elements](http://dicom.nema.org/medical/dicom/2020a/output/chtml/part06/chapter_6.html)

 
**why important?**
  
- DICOM can be seen as a universal way that medical imaging dataset are formatted
- Therefore, getting used to the **anatomy** and **data format** of DICOM makes it easier to
  -- dive into the data quicker
  -- know (or know how to find) the location where an information you want to retrieve is located (tag)
  -- know (or know how to find) the way the information is encoded/represented (value representation)

**hands-on!**

For example, if you want to know the age of the patient in the data:

1. You just type in "DICOM age" in Google and you can find that the tag is (0010,1010), and the **Value representation (VR)** is Age String (AS)
2. You then go to [this page](http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html), and you can see the definition of AS values
3. Given the definition written on the page, it seems that if the patient is 30 years old, the value will be 030Y


## 
