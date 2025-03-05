Folder structure:
--ID: store the csv file for loading dataset
  --csv file for labeled training dataset, unlabeled training dataset, val dataset and test dataset
  model_weights: save the model weights
  models: model architecture file
  utils: store the function needed. 
GRN_SEL.py: training for GRN-SEL
config_GRN_SEL.yaml: config file for GRN_SEL.py
GRN_SSL.py: training for GRN-SSL
config_GRN_SSL.yaml: config file for GRN_SSL.py
val_no_SGE: run inference on val/test dataset and print performance and confidence interval. It does not use segmentation guided Enhancement
validate_no_SGE.yaml: config file for val_no_SGE.py
val_SGE.py: run inference on val/test dataset and print performance and confidence interval. It implement segmentation guided Enhancement
validate_SGE.yaml: config file for val_SGE.py