# A Novel Personalized Federated Learning Method for Privacy-Preserving Smart Mobile Health Monitoring  

This is our implementation for the paper:  

**A Novel Personalized Federated Learning Method for Privacy-Preserving Smart Mobile Health Monitoring**.  




## Environment Settings  

- **Python 3.8.13**  
- **Required libraries**:  
  - `numpy`  
  - `pandas`  
  - `pytorch`  
  - `scikit-learn`  
  - *(etc.)*  

## Example to Run the Codes  

Run the following command to train the model on the PPMI dataset:  

```bash
python main.py --dataset ppmi --lr 0.01 --device cuda:1 --batch_size 128 --epoch 20 --client_frac 1 --model our
```

## Dataset
The three datasets used in our paper are all public datasets. You can download them through the following links:  

- **CrossCheck Dataset**  
  - Download: [https://www.mh4mh.org/eureka-data](https://www.mh4mh.org/eureka-data)  
  - Description: The CrossCheck dataset encompasses six categories of behaviors: physical behavior, dialogue behavior, mobile phone call behavior, displacement behavior, text message behavior, and mobile phone use behavior. 

- **Oxford Parkinson's Telemonitoring Dataset**  
  - Download: [https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)  
  - Description: The Oxford dataset focuses on the vocal characteristics of Parkinson’s disease (PD) patients, encompassing features such as pitch, volume, periodic variation, and nonlinear dynamics.

- **PPMI (Parkinson's Progression Markers Initiative) Dataset**  
  - Download: [https://www.ppmi-info.org/access-data-specimens/data](https://www.ppmi-info.org/access-data-specimens/data)  
  - Description:The PPMI dataset is a comprehensive medical database encompassing multimodal data, including clinical records, medical imaging, multi-omics analyses, genetic information, sensor monitoring data, and biomarker measurements. In this study, to align with mobile health monitoring scenarios, we specifically extracted physiological data of Parkinson’s disease patients collected via wearable devices. These data cover key dimensions such as motor function metrics, heart rate variability parameters, and sleep behavior patterns.
