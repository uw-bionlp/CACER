# CACER
Dataset and baseline experiments for the Clinical Concept Annotations for Cancer Events and Relations (CACER) dataset.

# Data preprocessing
## Dataset cleaning
Step 1. Automatic cleaning: preprocessing/automatic_cleaning.py
1. remove invalid arguments/triggers. 
2. merge the overlapping span of the same type
3. remove multiple trigger-argument links to the same (trigger, argument) pair

Step 2. Flag obvious descripancies to the entity annotation guideline, by labeling warnings in the BRAT .ann file and output a summary file. Need a human to resolve the disagreements: preprocessing/flag_entity_error.py
1. attributes not connected to any trigger (floating tbs)
2. labeled attributes without any labels
3. labeled attributes with multiple labels
4. multi labels for exact same span
5. problem trigger without assertion attributes
6. problem trigger with multiple assertion attributes

Step 3. Flag the same entity pair with multiple relations and output a summary file. Need a human to resolve the disagreements: preprocessing/flag_relation_error.py

## Dataset statistics
Using the script preprocessing/demographics.ipynb
1. Split the dataset into train, valid and test from the doubly and singly dataset.
2. Calculating the demographics

## Format conversion
1. Download the raw BRAT .ann file and store under the folder dataset/BRAT
2. Refer to the [brat scoring package](https://github.com/Lybarger/brat_scoring): copy the brat_scoring folder under the preprocessing folder.
3. Convert the BRAT .ann to json file through the program: preprocessing/ann2json.py
4. Convert the json format to other input formats for other experiments, using the script: preprocessing/format_convert.ipynb

# Experiments
## Spert
The code deploys the [SpERT](https://github.com/lavis-nlp/spert) package, please first clone corresponding directory and find the details from the script in experiments/spert/run.sh.

## PL-Marker
The code deploys the [PL-Marker](https://github.com/thunlp/PL-Marker) package, please first clone corresponding directory and the details from the script in experiments/plmarker/run.sh. Need to add the CACER dataset configuration to the main file.

## Flan-t5
The code deploys the huggingface [peft](https://github.com/huggingface/peft) package, and is adapted from this [post](https://www.philschmid.de/fine-tune-flan-t5-peft). Please find the details from the script in experiments/flan-t5/run.sh.

## llama
The code deploys the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) package, please first clone corresponding directory and find the details from the script in experiments/llama/run.sh.

# Evaluation


# Cite
