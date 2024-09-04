# CACER
Dataset and baseline experiments for the manuscript [CACER: Clinical concept Annotations for Cancer Events and Relations](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocae231/7748302?searchresult=1) dataset.

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
1. Split the dataset into train, valid, and test from the doubly and singly annotated dataset.
2. Calculating the demographics

## Dataset availability 
The dataset will be released after obtaining approval from the University of Washington (UW) Institutional Review Boards (IRBs).

## Format conversion
1. Download the raw BRAT .ann file and store it under the folder dataset/BRAT
2. Refer to the [brat scoring package](https://github.com/Lybarger/brat_scoring): copy the brat_scoring folder under the preprocessing folder.
3. Convert the BRAT .ann to JSON file through the program: preprocessing/ann2json.py
4. Convert the JSON format to other input formats for other experiments, using the script: preprocessing/format_convert.ipynb

# Experiments
## Spert
The code deploys the [SpERT](https://github.com/lavis-nlp/spert) package, please first clone the corresponding directory and find the details from the script in experiments/spert/run.sh.

## PL-Marker
The code deploys the [PL-Marker](https://github.com/thunlp/PL-Marker) package, please first clone the corresponding directory and the details from the script in experiments/plmarker/run.sh. Need to add the CACER dataset configuration to the main file.

## Flan-t5
The code deploys the huggingface [peft](https://github.com/huggingface/peft) package, and is adapted from this [post](https://www.philschmid.de/fine-tune-flan-t5-peft). Please find the details from the script in experiments/flan-t5/run.sh.

## llama
The code deploys the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) package, please first clone the corresponding directory and find the details from the script in experiments/llama/run.sh.

## Evaluation
The script for scoring and significance testing can be found under the folder, _evaluation_.

# Contact
For questions about this dataset, please contact Velvin Fu at velvinfu@uw.edu.

# Cite
```bibtex
@article{cacer2024fu,
    author = {Fu, Yujuan Velvin and Ramachandran, Giridhar Kaushik and Halwani, Ahmad and McInnes, Bridget T and Xia, Fei and Lybarger, Kevin and Yetisgen, Meliha and Uzuner, Özlem},
    title = "{CACER: Clinical concept Annotations for Cancer Events and Relations}",
    journal = {Journal of the American Medical Informatics Association},
    pages = {ocae231},
    year = {2024},
    month = {09},
    abstract = "{Clinical notes contain unstructured representations of patient histories, including the relationships between medical problems and prescription drugs. To investigate the relationship between cancer drugs and their associated symptom burden, we extract structured, semantic representations of medical problem and drug information from the clinical narratives of oncology notes.We present Clinical concept Annotations for Cancer Events and Relations (CACER), a novel corpus with fine-grained annotations for over 48 000 medical problems and drug events and 10 000 drug-problem and problem-problem relations. Leveraging CACER, we develop and evaluate transformer-based information extraction models such as Bidirectional Encoder Representations from Transformers (BERT), Fine-tuned Language Net Text-To-Text Transfer Transformer (Flan-T5), Large Language Model Meta AI (Llama3), and Generative Pre-trained Transformers-4 (GPT-4) using fine-tuning and in-context learning (ICL).In event extraction, the fine-tuned BERT and Llama3 models achieved the highest performance at 88.2-88.0 F1, which is comparable to the inter-annotator agreement (IAA) of 88.4 F1. In relation extraction, the fine-tuned BERT, Flan-T5, and Llama3 achieved the highest performance at 61.8-65.3 F1. GPT-4 with ICL achieved the worst performance across both tasks.The fine-tuned models significantly outperformed GPT-4 in ICL, highlighting the importance of annotated training data and model optimization. Furthermore, the BERT models performed similarly to Llama3. For our task, large language models offer no performance advantage over the smaller BERT models.We introduce CACER, a novel corpus with fine-grained annotations for medical problems, drugs, and their relationships in clinical narratives of oncology notes. State-of-the-art transformer models achieved performance comparable to IAA for several extraction tasks.}",
    issn = {1527-974X},
    doi = {10.1093/jamia/ocae231},
    url = {https://doi.org/10.1093/jamia/ocae231},
    eprint = {https://academic.oup.com/jamia/advance-article-pdf/doi/10.1093/jamia/ocae231/59003844/ocae231.pdf},
}
```


