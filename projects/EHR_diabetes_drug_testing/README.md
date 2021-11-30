# Expected Hospitalization Time with EHR Data

## Project Summary


## Dataset

This project use a synthetic dataset due to healthcare regulations (HIPPA, HITECH). The dataset was modified from the [UCI Diabetes dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008).

Details of each column in the dataset can be found in `project_data_schema.csv` on this [page](https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references). These are:

- `encounter_id` - encounter id
- `patient_nbr` - patient id
- `race` - (categorical) race: Caucasian, Asian, African American, Hispanic, and other
- `gender` - (categorical) gender: male, female, and unknown/invalid
- `age` - (categorical) age range: [0, 10), [10, 20), . . ., [90, 100)
- `weight` - weight in pounds
- `admission_type_id` - (categorical) 9 types of admission:
    - 1 - Emergency
    - 2 - Urgent
    - 3 - Elective
    - 4 - Newborn
    - 5 - Not Available
    - 6 - NULL
    - 7 - Trauma Center
    - 8 - Not Mapped
- `discharge_disposition_id` - (categorical) 29 types: discharged to home, expired, and not available
- `admission_source_id` - (categorical) 21 types: physician referral, emergency room, and transfer from a hospital
- `time_in_hospital` - (predictor) number of days between admission and discharge
- `payer_code` - (categorical) 23 types: Blue Cross/Blue Shield, Medicare, and self-pay
- `medical_specialty` - (categorical) 84 types: cardiology, internal medicine, family/general practice, and surgeon
- `primary_diagnosis_code` - primary diagnosis code (ICD9-CM)
- `other_diagnosis_codes` - 2 secondary diagnosis codes (ICD9-CM)
- `number_outpatient` - number of outpatient visits in the year preceding the encounter
- `number_inpatient` - number of inpatient visits in the year preceding the encounter
- `number_emergency` - number of emergency visits in the year preceding the encounter
- `num_lab_procedures` - number of lab tests during the encounter
- `number_diagnoses` - number of diagnoses enter to the system
- `num_medications` - number of distinct generic names administered during the encounter
- `num_procedures` - number of procedures (other than lab tests) performed during the encounter
- `ndc_code` - NDC code(s) for drug prescribed during encounter. Note that this field is denormalized
- `max_glu_serum` - (categorical) range of the result: ">200", ">300", "normal", and "none" if not measured
- `A1Cresult` - (categorical) range of the result:
    - ">8" if the result was greater than 8%
    - ">7" if the result was greater than 7% but less than 8%
    - "normal" if the result was less than 7%
    - "none" if not measured
- `change` - (categorical) indicator of change in diabetic medications: "change" or "no change"
