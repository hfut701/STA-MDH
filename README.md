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

Run the following command to train the CUFAIR model on the StackExchange dataset:  

```bash
python main.py --dataset ppmi --lr 0.01 --device cuda:1 --batch_size 128 --epoch 20 --client_frac 1 --model our
```

## Dataset
The three datasets used in our paper are all public datasets. You can download them through the following links:  

- **CrossCheck Dataset**  
  - Download: [https://www.mh4mh.org/eureka-data](https://www.mh4mh.org/eureka-data)  
  - Description: Contains annotated question-answer pairs from mental health forums, with expert-labeled usefulness scores and user comments. Used to train the model's comment-based relevance detection module.  

- **Oxford Parkinson's Telemonitoring Dataset**  
  - Download: [https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)  
  - Description: Includes biomedical voice measurements from Parkinson's patients. We utilized its structured metadata to validate the model's ability to handle heterogeneous data features.  

- **PPMI (Parkinson's Progression Markers Initiative) Dataset**  
  - Download: [https://www.ppmi-info.org/access-data-specimens/data](https://www.ppmi-info.org/access-data-specimens/data)  
  - Description: Longitudinal clinical and imaging data for Parkinson's research. Applied in our cross-domain experiments to test the framework's generalization capability.  
