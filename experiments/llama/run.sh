#conda activate llama_factory
## step: adding the dataset name to the llama factory config file: LLaMA-Factory/data/dataset_info.json

project='CACER_intra_sent' 
device=2
llama='meta-llama/Llama-2-7b-chat-hf'
trainset='R15_re_train'
testset="R15_re_test"  

CUDA_VISIBLE_DEVICES=${device} python src/train_bash.py \
    --stage sft \
    --model_name_or_path ${llama} \
    --do_train \
    --dataset ${trainset} \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir experiments/${project} \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --plot_loss \
    --fp16 
