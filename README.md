# Multimodal Attention for Alzheimer's Disease Classification
Code for the paper [Multimodal Attention-based Deep Learning for Alzheimer's Disease Diagnosis](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocac168/6712292).

## Dataset
We provide results on the [Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset](https://adni.loni.usc.edu/). The data is not provided in this repository and needs to be requested directly from ADNI.   

## Requirements:
Python 3.7.4 (and above)  
Tensorflow 2.6.0  
Further details on all packages used in this repository can be found in general/requirements.txt

## Description
In this work, we presented a multi-modal, multi-class, attention-based deep learning framework to detect Alzheimer's disease using genetic, clinical, and imaging data from ADNI.

<img src="https://user-images.githubusercontent.com/35315239/187262625-0f980b94-7cce-49ec-9041-421e56b67ecd.png" width="600">

This repository contains the code for the mentioned paper. The model architecture above is located in training/train_all_modalities.py. 

## Preprocessing
To create a list of patient IDs with their diagnosis, run the notebook general/diagnosis_making.ipynb. 

To preprocess clinical data run the notebook preprocess_clincal/create_clinical_dataset.ipynb, which will create a CSV file with the necessary data.

The CSVs used in the scripts above need to be obtained from ADNI directly and are thus not included with the notebooks. 

To preprocess imaging data, first run preprocess_images.py with the directory where images are stored as the argument. Then, use the file created from the script to run the notebook in preprocess_images/splitting_image_data.ipynb to split your data into training and testing.

To preprocess genetic data (SNPs), first obtain VCF files from ADNI. Then use the vcftools package to filter the files based on your chosen criteria (Hardy-Weinberg equilibrium, genotype quality, minor allele frequency, etc.). To further filter the VCF files according to the AD-related genes from AlzGene Database (http://www.alzgene.org/), run filter_vcfs.py script. Next, to compile all the genetic files together run concat_vcfs.py. Finally, to further reduce the number of features, run the notebook create_genetic_dataset.ipynb. All scripts can be found in the preprocess_genetic folder. 

## Training and Evaluation

To train and evaluate a uni-modal model baseline, run train_clinical.py, train_genetic.py, or train_imaging.py.
To train and evaluate the multimodal architecture, run train_all_modalities.py.



## Credits

Some of the structure in this repo was adopted from https://github.com/soujanyaporia/contextual-multimodal-fusion

## Authors

[Michal Golovanevsky](https://github.com/michalg04)
