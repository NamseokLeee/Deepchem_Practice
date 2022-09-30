# Deepchem_Practice
Simple fully connected neural network models estimating drug efficacy against HIV and drug water solubility.  
Based on molecular fingerprint featurization by Deepchem library and Tensorflow DNN models.  

## Environment
- Python = 3.7
- Deepchem = 2.6.0
- Tensorflow = 2.2.0

## Results
### HIV prediction:  
![1](https://user-images.githubusercontent.com/90392853/193253529-92fd4660-e377-4094-834b-079d7946821b.GIF)  
Accuracy(random split): **0.971**  
ROC-AUC(random split):  **0.799**  

### Water solubility estimation:
RMSE: **1.194**  
R squared: **0.738546**  


## References
HIV dataset:  
*AIDS Antiviral Screen Data.*  
https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data

ESOL dataset:  
*Delaney, John S. "ESOL: estimating aqueous solubility directly from molecular structure." Journal of chemical information and computer sciences 44.3 (2004): 1000-1005.*
