# Compressed-Bio_ClinicalBER

**MIMIC-III, Mayo Clinic Bipolar Disorder, and Quebec Congenital Heart Disease EHR datasets**


**Dataset**: https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k

**CITATION**

Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi,
Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark.**Mimic-iii, a freely accessible
critical care database**. Scientific Data, 3:160035, 2016.

**Requirement**


**Symptom Extraction Instructions (ClinicalBERT Multiclass Version)**

1. Install Dependencies
Ensure you have Python ≥3.8 and install required packages:

pip install transformers datasets scikit-learn pandas torch
If you're using Jupyter  or Colab on AWS or GCP instance, run:
!pip install transformers datasets scikit-learn pandas torch

 2. Prepare Your Dataset
You should have a file named:

NOTEEVENTS_random.csv (sampled subset of MIMIC-III clinical notes)

This file must contain a column:

"TEXT" — a string of clinical notes.

The code extracts a single symptom label per document by checking whether one of the following symptoms is present:

symptom_list = ["edema", "pain", "cough", "fever", "bleeding"]
The model assigns a unique class (label) based on the first matching symptom found in the text.

 3. Data Preprocessing and Labeling
Your script:

Converts text to lowercase

Checks for presence of symptoms

Assigns a label index (0 to 4) depending on the first matched symptom

Drops samples with no matching symptoms

No manual annotation is required!

 4. Model Training
The code fine-tunes five versions of emilyalsentzer/Bio_ClinicalBERT:

Version	Description
Base	Standard Bio_ClinicalBERT
Pruned	Linear layers sparsified by 30%
Low-Rank	Linear layers decomposed with SVD (rank=32)
Quantized	Dynamically quantized model (int8 weights)
Distilled	Knowledge distillation from teacher model

Each model:

Trains for 5 epochs

Uses CrossEntropyLoss for multiclass classification

Tracks metrics per epoch: accuracy, precision, recall, F1, AUROC

Saves metrics to .csv (e.g. multiclass_base_metrics.csv)

 5. Evaluation
After each epoch, the model is evaluated on a held-out test set using:

Accuracy

Macro Precision, Recall, F1

AUROC (One-vs-Rest for multiclass)

Metrics are printed and also saved in CSV format for later comparison.

▶ 6. How to Run Everything
To launch all experiments sequentially, run:

use the function run_all() which executes the following sub-function:


base_model()
pruning_model()
lowrank_model()
distillation_model()
quantization_model()

7. Check Results
Each model’s evaluation results are written into a CSV file:

multiclass_(technique)_metrics.csv


