# Prompt Pool based Class-Incremental Continual Learning for Dialog State Tracking
This is the official code for "Prompt Pool based Class-Incremental Continual Learning for Dialog State Tracking" (ASRU2023). This code is based on [CPT4DST](https://github.com/thu-coai/CPT4DST). **The package requirements and the dataset processing is the same as CPT4DST**.
## Experiments
The following script will sequentially execute model training and evaluation.
```
python prompt_pool_tuning.py \
    --train data/sgd_train.json \
    --dev data/sgd_dev.json \
    --test data/sgd_test.json \
    --schema data/all_schema.json \
    --select_method bert-encoder \
    --epochs 3\
    --learning_rate 2e-5 \
    --batch_size 8 \
    --gpu_id 0 \
    --dataset_order 1 \ # The order of training tasks
    --top_n 10 \ # the number of selected prompts for each task
    --pool_size 150\ # the number of prompts in the prompt pool
    --penalty \ # whether to add penalty items
    --M 50 \ # the rehearsal buffer size
    --dis_method euclidean\ # the method of calculating distance
    --model_name t5-small \ # the backbone, can be chosen from 'google/mt5-small', 't5-small', 't5-base', 't5-large'
    --dataset sgd \ # the dataset, sgd or CM-Pickup (not released currently)
    --memory_type fix_size \ # the storage method for rehearsal buffer, fix_size (maintain a fixed buffer) or incre_size (store the same number of samples for each task)
```
* Please contact liuhong21@mails.tsinghua.edu.cn if you have any questions.
