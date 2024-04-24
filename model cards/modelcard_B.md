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
The proposed model is an ensemble of BiLSTMs with Attention mechanisms. It aims to do pairwise sequence classification and was trained for the Natural Language Inference (NLI) task to detect whether a sequence (hypothesis) is true based on another sequence (premise).

<!-- ... -->


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
The proposed model is an ensemble of multiple BiLSTMs with Attention mechanisms. Note that each BiLSTM within the ensemble is refered to as a base learner.

For each base learner, RoBERTa word embeddings are employed to represent the hypothesis-premise sequences as input to the BiLSTM, followed by a self-attention layer. Then, a dense layer with a single neuron using sigmoid activation calculates the probability of the prediction being True (1).

To generate the final prediction, the probabilities from each base learner are combined using a normalized geometric mean combiner. For further information on this combiner, please refer to [this paper]("https://arxiv.org/abs/2301.03962").


<!-- ... -->

- **Developed by:** Samuel Belkadi and Frenciel Anggi
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Ensemble of BiLSTM with Atttention mechanisms
- **Finetuned from model [optional]:** NA

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** NA
- **Paper or documentation:** NA

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

Each base learner was trained on 25,944 out of 26,944 pairs of hypothesis-premise sequences from the provided NLI training set. 1,000 samples were disregarded due to memory constraints. Data pre-processing steps involved: (1) removal of empty hypotheses, (2) tokenization of sequence pairs, and (3) pre-extraction of word embeddings from `roberta-base`. This pre-extraction step was implemented to exclude RoBERTa's embedding layer from each learner's architecture, saving both computational resources and inference time during forward passes.

<!-- [More Information Needed] -->

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
Each input sequence was truncated to 150 tokens, in which each token were embedded using the `roberta-base` pretrained model. These embedded sequences are then fed into each base learner within the ensemble.

For each base learner, the input layer accepts the word embeddings, passing them through a BiLSTM layer. The resulting outputs are then processed by a `SeqSelfAttention` layer from the `keras_self_attention` library, utilizing a sigmoid function for attention activation. A dropout layer is subsequently applied before the outputs pass through a final dense layer comprising a single neuron with a sigmoid activation function, predicting the probability of the prediction being True (1).

Finally, the predictions of each base learner are combined using a geometric mean combiner.

Each learner are trained independently with the Adam optimizer and binary cross-entropy loss function. Although bagging or boosting may have been the best ensemble methods, we were unable to employ them due to memory constraints. 

During hyperparameter tuning, a Grid Search approach was utilized to adjust the ensemble size from 1 to 25 base learners, while the number of epochs was determined via Early Stopping capped at 8 epochs. A distinct validation set of about 6K sequence pairs were used to select the best set of hyperparameters and perform Early Stopping.

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->
The following hyperparameters were set during training:
* sample_size 	= 150
* embedding_dim = 768 (following `roberta-base` embedding dimension)
* n_lstm 		= 128
* dropout		= 0.2
* optimizer		= Adam
* batch_size	= 12

And the following hyperparameters were selected after hyperparameter tuning:
* epochs        = 8
* n_learners    = 11

<!-- [More Information Needed] -->

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

* Overall Ensemble Training Time: 1 hour 7 minutes
* Duration per training epoch (per base learner): 45 seconds 
* Duration per training epoch (for entire ensemble): 8 minutes 25 seconds
* Model Size: 
    - Ensemble          : 10,100,000 parameters, 38.5 MB
    - One base learner  : 918,606    parameters, 3.50 MB

<!-- [More Information Needed] -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->
The entire development/validation set provided, amounting to ~6K pairs, was used to evaluate the model.

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
| f1 (weighted)     |   0.962  | 0.802 |
| f1 (macro) 		|   0.962  | 0.802 |
| precision (macro) |   0.962  | 0.804 |
| recall (macro) 	|   0.963  | 0.803 |
| accuracy        	|   0.963  | 0.802 |
| roc (macro)		|   0.962  | 0.803 |

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
* tensorflow==2.14.0 
* keras-self-attention 
* transformers  
* accelerate
* scikit-learn 

<!-- [More Information Needed] -->

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The presence of imbalanced classes in the training data can introduce bias toward the majority class, potentially leading to misclassification or underrepresentation of the minority class. Despite precautions taken to prevent overfitting during training, the model may still exhibit bias toward the majority class.

Moreover, this model's training is constrained by the hardware and resources available, preventing the utilization of the entire training dataset (26,944 pairs). Additionally, memory constrains restricted the implementation of preferred ensemble methods such as bagging and boosting.

Finally, the ensemble model employed strong base learners (BiLSTMs), whereas weak learners are typically preferred.

<!-- [More Information Needed] -->

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

<!-- [More Information Needed] -->
