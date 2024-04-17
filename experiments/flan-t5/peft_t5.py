import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from peft import PeftModel, PeftConfig
from datasets import load_dataset, concatenate_datasets,Features,Value, Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType,PeftModelForCausalLM
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl, pipeline,AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from torch.utils.data import DataLoader
# Fine-Tune Llama2-7b on SE paired dataset
from dataclasses import dataclass, field
from typing import Optional
import time
import pandas as pd
import json
from tqdm import tqdm
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import json

def get_args():
    parser = argparse.ArgumentParser(description='Process some configurations.')
    
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the validation data')

    parser.add_argument('--model_id', default="google/flan-t5-large", type=str, required=True, help='Name of the model')
    parser.add_argument('--num_epoch', default=15, type=int, required=True, help='Number of epochs')
    parser.add_argument('--checkpoint', default=None, type=int, help='check point to evaluate')
    parser.add_argument('--experiment_name', default="temp",type=str, required=True, help='experiment name')
    parser.add_argument('--mode', default="train",type=str, help='train or eval')
    

    parser.add_argument('--cuda_device', default='1', type=str,  help='CUDA device ID')

    parser.add_argument('--max_target_length', default=50, type=int,  help='max generation length for the target sequence')
    parser.add_argument('--hf_acceess_token', default='', type=str,  help='acceess token to access the T5 model')


    parser.add_argument("--split", type=str, default="train", help="the split to use")
    parser.add_argument("--size_valid_set", type=int, default=10000000, help="the size of the validation set")
    parser.add_argument("--streaming", type=bool, default=True, help="whether to stream the dataset")
    parser.add_argument("--shuffle_buffer", type=int, default=5000, help="the shuffle buffer size")
    parser.add_argument("--seq_length", type=int, default=1024, help="the sequence length")

    parser.add_argument("--max_steps", type=int, default=500, help="the maximum number of sgd steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="the logging frequency")
    parser.add_argument("--save_steps", type=int, default=10, help="the saving frequency")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="the per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="the per device eval batch size")
    parser.add_argument("--per_device_test_batch_size", type=int, default=128, help="the per device test batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="the gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="whether to use gradient checkpointing")
    parser.add_argument("--group_by_length", type=bool, default=True, help="whether to group by length")

    parser.add_argument("--lora_alpha", type=float, default=16, help="the lora alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="the lora dropout parameter")
    parser.add_argument("--lora_r", type=int, default=64, help="the lora r parameter")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="the learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="the lr scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="the number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="the weight decay")
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit", help="the optimizer type")

    parser.add_argument("--log_freq", type=int, default=1, help="the logging frequency")

    args = parser.parse_args()

    return args

def concatenate_columns(example):
    example['all_input'] = example['instruction'] + "\n" + example['input']
    if 'options' in example:
        example['all_input'] += example['options']
    return example

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    if 'options' in sample:
        inputs = [ instruct + "\n" + inpt + opt for inpt, instruct , opt in zip(sample["input"],sample["instruction"],sample['options'])]
        print(inputs)
    else:
        inputs = [ instruct + "\n" + inpt  for inpt, instruct in zip(sample["input"],sample["instruction"])]
    
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_model(args):
    if args.checkpoint is not None or args.mode=='eval':
        # loading from checkpoint
        if args.mode=='eval' and args.checkpoint is None:
            model_dir=f"models/{args.experiment_name}"
        else:
            model_dir=f"models/{args.experiment_name}/checkpoint-{args.checkpoint}" 
        
        print('loading checkpoint', model_dir)
        config = PeftConfig.from_pretrained(model_dir)
        # load base LLM model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id,  
                                                    load_in_8bit=True,
                                                    device_map={'':torch.cuda.current_device()})
        
        # Load the Lora model
        model = PeftModel.from_pretrained(model,model_dir, device_map={"":0})
    else:
        # load model from the hub
        if args.model_id=="luqh/ClinicalT5-large":
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, device_map="auto",from_flax=True)

        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, load_in_8bit=True, 
                                                        device_map={'':torch.cuda.current_device()},
                                                        torch_dtype=torch.float32)
    return model

def evaluate_peft_model(sample):
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), 
                             do_sample=False, top_p=0.9, 
                             max_new_tokens=max_target_length)
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # Some simple post-processing
    return prediction, sample['id']

def evaluate_peft_model_batch(sample):
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].cuda(), 
                             do_sample=False, #top_p=0.9, 
                             max_new_tokens=max_target_length)
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Some simple post-processing
    return results
if __name__ == "__main__":
    
    args=get_args()

    os.makedirs('models',exist_ok=True)
    os.makedirs('predictions',exist_ok=True)

    # loading the dataset
    data_path={
                'train':args.train_path, 
                'eval':args.valid_path
                }
    dataset=load_dataset("json", data_files=data_path)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"eval dataset size: {len(dataset['eval'])}")

    ## tokenizing the dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["eval"]])\
                                                                .map(concatenate_columns)\
                                                                .map(lambda x: tokenizer(x["all_input"], truncation=True), \
                                                                batched=True, remove_columns=["input", "output","instruction"])
    input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
    max_source_length = int(np.percentile(input_lenghts, 97))
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["eval"]]).map(lambda x: tokenizer(x["output"], truncation=True), batched=True, remove_columns=["input", "output","instruction"])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # take 90 percentile of max length for better utilization
    max_target_length = int(np.percentile(target_lenghts, 100))
    print(f"Max target length: {max_target_length}")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input", "output","instruction"])

    # save datasets to disk for later easy loading
    tokenized_dataset["train"].save_to_disk(f"data/{args.experiment_name}_train")
    tokenized_dataset["eval"].save_to_disk(f"data/{args.experiment_name}_test")


    if args.mode=='train':
        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        model=load_model(args)
        #model
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # we want to ignore tokenizer pad token in the loss
        label_pad_token_id = -100
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )


        # Define training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"models/{args.experiment_name}",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps, 
            learning_rate=args.learning_rate, 
            num_train_epochs=args.num_epoch,
            logging_dir=f"models/{args.experiment_name}/logs",
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy="epoch", #steps
            save_steps=args.save_steps,
            report_to="tensorboard",
            overwrite_output_dir=True,
            # evaluation_strategy='steps',
            # fp16=True,
        )

        # Create Trainer instance
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["eval"]
        )
        model.config.use_cache = False  

        # train model
        trainer.train()

        os.makedirs(f"models",exist_ok=True)
        # Save our LoRA model & tokenizer results
        trainer.model.save_pretrained(f"models/{args.experiment_name}")
        tokenizer.save_pretrained(f"models/{args.experiment_name}")
        
        
        df=pd.DataFrame(trainer.state.log_history)
        df.to_csv(f"models/{args.experiment_name}/loss.csv")
    
    else:
        model=load_model(args)
        model.eval()

        # load test dataset from distk
        test_dataset = load_from_disk(f"data/{args.experiment_name}_test").with_format("torch")
        os.makedirs(f"predictions/{args.experiment_name}",exist_ok=True)
        if args.checkpoint is not None:
            output_file=f"predictions/{args.experiment_name}/cpt{args.checkpoint}_predictions.json"
        else:
            output_file=f"predictions/{args.experiment_name}/predictions.json"
        if os.path.isfile(output_file):
            predictions=json.loads(open(output_file).read())
        else:
            predictions={}
    

        test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_test_batch_size, shuffle=False)
        #batch=[]
        for sample in tqdm(test_dataloader):
            if all([id in predictions for id in sample['id']]):
                continue
            preds = evaluate_peft_model_batch(sample)
            for id,pred in zip(sample['id'],preds):
                predictions[id]=pred
            assert len(preds)==len(sample['id']),preds

            with open(output_file,"w") as f:
                json.dump(predictions,f,indent=4)

        print(output_file)
