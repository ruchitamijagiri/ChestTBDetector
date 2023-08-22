# Chest Tuberculosis Detection using Deep Learning in OpenMRS

This project aims to build an AI system for automatic detection of tuberculosis from chest X-ray images using deep convolutional neural networks. It is integrated into OpenMRS by creating an Open Web App (OWA).

## Dataset

The model is trained on the NIAID TB Portals X-ray dataset containing 1510 images positives for tuberculosis and 4082 normal chest X-ray images. The data is balanced using SMOTE.

## Model Architecture

The model uses a DenseNet-121 CNN architecture pre-trained on ImageNet weights. The convolutional base is frozen and the classification head re-trained for TB classification.

## Usage

The Jupyter notebooks provide the model training code and inference on test data. The final model is serialized in .h5 format. A Flask app serves predictions using the trained model.

The frontend allows uploading images and viewing model predictions with confidence scores.

Steps
1. Install OpenMRS standalone version.
2. The app folder present in 'ChestTBdetector_OWA' needs to be zipped and uploaded into the installed OpenMRS.
3. Create a ChestTB detection button on the home screen of the installed OpenMRS following the steps on https://wiki.openmrs.org/pages/viewpage.action?pageId=80380697

## Results

The model achieves 71% validation accuracy on TB Portals test data.
