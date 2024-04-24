---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/SamySam0/nlu-cwk-nli

---

# Model Card for n61655sb-j49970fa-NLI

<!-- Provide a quick summary of what the model is/does. -->
The proposed model is a large transformer. It aims to do pairwise sequence classification and was trained for the Natural Language Inference (NLI) task to detect whether a sequence (hypothesis) is true based on another sequence (premise).

<!-- ... -->


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
The proposed model is based on the pretrained RoBERTa-Large transformer, which we fine-tuned on ~26K pairs of hypothesis-premise sequences using the Low-Rank Adaptation (LoRA) method [https://arxiv.org/abs/2106.09685].

LoRA is a technique designed to fine-tune very large language models by keeping the pretrained parameters of the model frozen and introduce trainable low-rank matrices that adapt the model's behavior for a specific task. This significantly reduces the number of trainable parameters during fine-tuning, leading to faster training and reduced computational costs. In a typical transformer architecture, attention and feed-forward layers play crucial roles. Therefore, LoRA specifically targets the weight matrices in these layers.

<!-- ... -->

- **Developed by:** Samuel Belkadi and Frenciel Anggi
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** roberta-large

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** [https://github.com/facebookresearch/fairseq/tree/main/examples/roberta]
- **Paper or documentation:** [https://arxiv.org/abs/1907.11692]

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->
The low-rank matrices were trained on all 26,944 pairs of hypothesis-premise sequences from the provided NLI training set, and a distinct validation set of about 6K sequence pairs were used to select the best set of hyperparameters and perform Early Stopping.

Data pre-processing steps involved: (1) removal of empty hypotheses and (2) tokenization of sequence pairs with RoBERTa's tokenizer.


<!-- [More Information Needed] -->

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
By default, RoBERTa-Large has over 355 million parameters and takes about 22 hours to train on high-end hardware configurations. Therefore, as we cannot afford such requirements, we decided to accelerate training and reduce resource needs by employing the Low-Rank Adaptation technique (LoRA).

To this end, we freeze the pretrained weights of RoBERTa-Large and introduce an additional 1,775,000 parameters, as trainable low-rank matrices, within the model. Consequently, only about 0.5% of the model's parameters remain to be trained, saving approximately 91% of training time and resources.


#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

The following hyperparameters were set during training:
* optimizer 	= AdamW
* seed 			= 42 (for reproducibility)
* fp16 			= True
* batch_size 	= 8

The hyperparameter space used during hyperparameter tuning was:
* weight_decay 	= [0.01, 0.02]
* learning_rate = [5e-5, 3e-5, 2e-5]
* gradient_accumulation_steps = [1, 2, 3]

Additionally, the number of epochs were selected through Early Stopping, capped to 10 epochs with patience = 2.
Model selection and Early Stopping were decided based on the validation loss.

The following hyperparameters were selected after hyperparameter tuning:
* weight_decay 	= 0.02
* learning_rate = 5e-5
* gradient_accumulation_steps = 3
* batch_size 	= 8
* num_epochs 	= 7

<!-- [More Information Needed] -->

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

* Training Speed: 2 hours 7 minutes (instead of 22 hours without LoRA)
* Duration per training epoch: 18 minutes
* Model Size: 
	- LoRA matrices (trained)	: 1,750,000   parameters, 10 MB
	- Pretrained model (frozen)	: 355,000,000 parameters, 2.24 GB
	- Total						: 372,750,000 parameters, 2.24 GB

<!-- [More Information Needed] -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->
The entire development/validation set provided, amounting to ~6K pairs, is used to evaluate the model.

<!-- [More Information Needed] -->

#### Metrics

<!-- These are the evaluation metrics being used. -->

- F1 (Macro, Weighted)
- Precision (Macro)
- Recall (Macro)
- Accuracy
- ROC (Macro)

Note: We use macro averages to account for imbalanced classes.

<!-- [More Information Needed] -->

### Results

| metric            | training | dev   |
| :---------------- | :------: | :---: |
| f1 (weighted)     |   94.02  | 91.60 |
| f1 (micro)    	|   94.02  | 91.60 |
| f1 (macro) 		|   94.01  | 91.58 |
| precision (macro) |   94.06  | 91.62 |
| recall (macro) 	|   93.98  | 91.56 |
| accuracy        	|   94.02  | 91.60 |
| roc (macro)		|   93.98  | 91.56 |

<!-- [More Information Needed] -->

## Technical Specifications

### Hardware

The training process was performed on a Linux Centos virtual machine providing:
- 64GB RAM, 
- two 2.60GHz 12-core processors (Intel Xeon E5-2690 v3), and 
- a Tesla V100 GPU. 

The virtual machine was issued by the Computational Shared Facility (CSF3) of the University of Manchester.

<!-- [More Information Needed] -->

### Software

* datasets
* torch
* transformers  
* peft
* accelerate -U
* scikit-learn 
* wandb

<!-- [More Information Needed] -->

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The presence of imbalanced classes in the training data can introduce bias toward the majority class, potentially leading to misclassification or underrepresentation of the minority class. Despite precautions taken to prevent overfitting during training, the model may still exhibit bias toward the majority class.

<!-- [More Information Needed] -->

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

<!-- [More Information Needed] -->
