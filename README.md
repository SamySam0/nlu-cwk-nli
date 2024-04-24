# COMP34812 NLU Coursework: Natural Language Inference (NLI) Shared Task

Our group have chosen the Natural Language Inference (NLI) track, in which:
Given a premise and a hypothesis, determine if the hypothesis is true based on the premise. 

We have developed two different solutions to this problem on the following categories:

    (b) Deep learning-based approaches that do not employ transformer architectures 
    (c) Deep learning-based approaches underpinned by transformer architectures

Which have been developed in two separate (.ipynb) notebooks:

(b) `ensemble.ipynb`, and

(c) `lora.ipynb`

and their demo code on the following notebooks:

(b) `ensemble-demo.ipynb`, and

(c) `lora-demo.ipynb`

## Category B: Ensemble of BiLSTMs

### Development Code (Training and Evaluation)

The code for this category in `ensemble.ipynb` is sectioned into several parts: 
1. __Setup__: Libraries, dependencies and necessary functions are installed and imported, paths and filenames are set according to the user's configuration.
2. __Data Preparation__: Data are pre-processed, tokenized, and represented into RoBERTa's word embeddings.
3. __Ensemble Preparation (Model Initialisation)__: BiLSTM with attention mechanism are constructed and initialised as base learners. Function to combine predictions as an ensemble model are defined.

4. __Evaluation Preparation__: Define functions to 
    (1) calculate evaluation metrics and 
    (2) evaluate predictions. 

5. __Hyperparameter Tuning & Training__: Run hyperparameter tuning (Grid Search) and train models.
6. __Evaluation__: evaluates performance metrics on validation data using the best ensemble model.

### Demo Code

The final trained models for this category are saved and stored in OneDrive in [this link](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/frenciel_anggi_student_manchester_ac_uk/EvcPzYsuwfVFv4PPlNN2SvIB-abZeTyjG9rUmLdMs76iKQ): 

https://livemanchesterac-my.sharepoint.com/:f:/g/personal/frenciel_anggi_student_manchester_ac_uk/EvcPzYsuwfVFv4PPlNN2SvIB-abZeTyjG9rUmLdMs76iKQ

where anyone in the University of Manchester can view, access, and download it.

Please change the model paths in the **Setup** section in `ensemble-demo.ipynb` to refer to the models in OneDrive. You could download the models to your local machine and change the paths appropriately following your preference, for example:

```
# Path to all models of the ensemble
ENSEMBLE_PATH = './ensemble/models'

# Path to dataset in which the test file (xxx.csv) is located
DATASET_PATH = './data'
TEST_FILE_NAME = 'test.csv'

# Path to save the predictions
PREDICTION_PATH = './ensemble'
```

The following sections in this notebook: **Data Preparation** and **Predict with ensemble of models** can be run comfortably to prepare the test data, load the saved model and predict results given the test data.

Optionally, an evaluation section was provided at the very end to evaluate predictions if labels are available.

## Category C: LoRA

### Development Code (Training and Evaluation)

The code for this category in `lora.ipynb` is sectioned into several parts: 
1. __Setup__: Libraries, dependencies and necessary functions are installed and imported, paths and filenames are set according to the user's configuration.
2. __Data Preparation__: Data are pre-processed, tokenized, and finally collated into batches .
3. __LoRA Preparation (Model Initialisation)__: Loads a pre-trained model and setup LoRA (Low-Rank Adaptation) to speed up training and lower computational costs.

    LoRA is a technique designed to fine-tune very large language models by keeping the pretrained parameters of the model frozen and introduce trainable low-rank matrices that adapt the model's behavior for a specific task. This significantly reduces the number of trainable parameters during fine-tuning, leading to faster training and reduced computational costs.
        
    In a typical transformer architecture, attention and feed-forward layers play crucial roles. Therefore, LoRA specifically targets the weight matrices in these layers.

    Publication: https://arxiv.org/abs/2106.09685 

4. __Fine-tuning Preparation__: Define functions to 

    (1) calculate evaluation metrics and 

    (2) fine-tune the low-rank matrices (layers).

5. __Hyperparameter Tuning & Training__: Run hyperparameter tuning (Grid Search) and train model.
6. __Evaluation__: evaluates performance metrics on validation data using the best ensemble model.

### Demo Code

The models for this category are stored in OneDrive in [this link](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/frenciel_anggi_student_manchester_ac_uk/EvcPzYsuwfVFv4PPlNN2SvIB-abZeTyjG9rUmLdMs76iKQ): 

https://livemanchesterac-my.sharepoint.com/:f:/g/personal/frenciel_anggi_student_manchester_ac_uk/EvcPzYsuwfVFv4PPlNN2SvIB-abZeTyjG9rUmLdMs76iKQ

where anyone from the University of Manchester can view, access, and download it.

Please change the model paths in the **Setup** section in `lora-demo.ipynb` to refer to the models in OneDrive. You could download the models to your local machine and change the paths appropriately following your preference, for example:

```
# Fine-tuned model path (best model)
FT_MODEL_PATH = './LoRA/Final-model'

# Path to dataset in which the test file (xxx.csv) is located
DATASET_PATH = './data'
TEST_FILE_NAME = 'test.csv'

# Path to save the predictions
PREDICTION_PATH = './LoRA'
```

The following sections in this notebook: **Data Preparation** and **Predict with LoRA model** can be run comfortably to prepare the test data, load the saved model and predict results given the test data.

Optionally, an evaluation section was provided at the very end to evaluate predictions if labels are available.
