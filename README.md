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
1. Split the dataset into train, valid and test from the doubly and singly dataset.
2. Calculating the demongraphics

## format conversion
1. download the raw BRAT .ann file and store under the folder dataset/BRAT
2. 
