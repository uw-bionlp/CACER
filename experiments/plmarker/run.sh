# conda activate plmarker

## named entity recognotion
python3  run_acener.py  \
    --model_type bertspanmarker  \
    --model_name_or_path  /home/velvinfu/models/Bio_ClinicalBERT   \
    --data_dir R15  \
    --learning_rate 2e-5  \
    --num_train_epochs 15  \
    --per_gpu_train_batch_size  8  \
    --per_gpu_eval_batch_size 4  \
    --gradient_accumulation_steps 4  \
    --eval_all_checkpoints  \
    --max_seq_length 512 \
    --save_steps 1000  \
    --save_total_limit 5 \
    --max_pair_length 128  \
    --max_mention_ori_length 8  \
    --do_eval --do_train --evaluate_during_training  \
    --seed 42  \
    --onedropout  --lminit  \
        --train_file ../dataset/plmarker/train_plmarker.jsonl \
        --dev_file ../dataset/plmarker/valid_plmarker.jsonl \
        --test_file ../dataset/plmarker/test_plmarker.jsonl  \
        --output_dir experiments/CACER_ner  \
    --output_results --overwrite_output_dir

python3  run_re.py  --model_type bertsub  \
        --model_name_or_path  emilyalsentzer/Bio_ClinicalBERT   \
        --data_dir R15  \
        --learning_rate 2e-5  \
        --num_train_epochs 15 \
        --save_total_limit 10 \
        --per_gpu_train_batch_size  4  \
        --per_gpu_eval_batch_size 2  --gradient_accumulation_steps 16  \
        --max_seq_length 512  --max_pair_length 128  --save_steps 5000  \
        --save_results \
        --do_eval  --do_train --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
        --seed 42    \
        --use_ner_results \
        --train_file ../dataset/plmarker/train_plmarker.jsonl \
        --dev_file ../dataset/plmarker/valid_plmarker.jsonl \
        --test_file ../dataset/plmarker/test_plmarker.jsonl  \
        --output_dir experiments/CACER_re  \
        --overwrite_output_dir