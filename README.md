# GUI Neuron Labeling Tool

This Graphical User Interface Labeling Tool was devoloped to combine extracted neuron labels from different extraction methods (PCA-ICA and CNMF-E). 

# Documentation
For a detailed documentation of the GUI Labeling Tool refer to the documentation provided in the file GUI_Documentation.pdf. 

# Data
As the data files are too big please contact carlotta.ruppert@gmail.com for receiving the following file structure: 

For now the calcium imaging data needs to be provided both as an .h5 and .avi file. Test data is provided in the folder data and contains the following files:
- preprocessedMovie.h5
- preprocessedMovie5.avi

- annotation files for pca-ica:
    - dataPackedForGeneration.mat
    - cellMap.mat
    - resultsPCAICA.mat
    
- annotation files for CNMF-E
    - binary_masks.mat
    - cellmap.mat
    - traces.mat
    
 The tool was also used to quantify the results using a synthetic data set which is accesable as well in the subfolder data/synthetic/.
 
 # Data Conversion
 Files used for conversion of the annotations:
 - cnmfe_get_results.m extraction file for the CNMF-E method ( https://github.com/zhoupc/CNMF_E ) 
 - Convert_GUI_mat_2_UNET.ipynb file for converting annotations such that they can be loaded in the 3d U-Net ( https://github.com/VoytechG/neuron-finder )

# Evaluation
Code used for evaluation of the improved GUI labels can be found under /evaluation/Metrics and synthetic data eval.ipynb 
For a more detailed explanation of the evaluation metrics refer to the file Documentation_Evaluation_Metrics.pdf.

The tool was developed on Linux Mint 20.
