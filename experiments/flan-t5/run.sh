
# conda activate peft
# training
model_name='google/flan-t5-large'


### intra-sentence relation extraction
path_for_the_train_set='../../dataset/GenQA/train_re.json'
path_for_the_validation_set='../../GenQA/test_re.json'
experiment_type='CACER_re_intro'
num_epoch=30

python peft_t5.py \
                      --train_path ${path_for_the_train_set} \
                      --valid_path ${path_for_the_validation_set} \
                      --model_id ${model_name} \
                      --num_epoch ${num_epoch} \
                      --experiment_name ${experiment_type} \
                      --per_device_test_batch_size 256 \
                      --mode 'train' 