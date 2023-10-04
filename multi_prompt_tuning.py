import os
import json
import argparse
import numpy as np
import torch
import re
from load_data import DST_Dataset, DST_prompt_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup

from tqdm import tqdm
import wandb

from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer
from prompt_files.prompts_config import UNUSED_TOKENS, PROMPT_TOKENS, META_PROMPT_TOKENS

wandb.init(project="multi-task prompt tuning")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cal_acc(pre, label, slot_num):

    dst_predictions = [re.sub('\<(extra_id_)(.*?)\>', '', s) for s in pre]
    dst_predictions = [_.strip() for _ in dst_predictions]
    acc = 0
    for i, pred_str in enumerate(dst_predictions):
        new_pre = ''
        for j in range(slot_num[i]):
            left_token = '<extra_id_{}>'.format(j)
            right_token = '<extra_id_{}>'.format(j + 1)
            if right_token not in pred_str:
                right_token = '</s>'
            try:
                value = pred_str.split(left_token)[1].split(right_token)[0].strip()
            except:
                value = 'NONE'
            new_pre += '<extra_id_{}>'.format(j) + value
        if new_pre == label[i]:
            acc += 1
    return acc

def main(args):
    
    device = torch.device("cuda:{}".format(args['gpu_id']) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    wandb.config.update(args)
    
    if args["dataset_order"] == 1:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
    elif args["dataset_order"] == 2:
        dataset_order = ["['sgd_hotels_4']", "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']",
                         "['sgd_media_2']", "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_trains_1']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_flights_1']",
                         "['sgd_services_4']", "['sgd_homes_1']", "['sgd_hotels_1']"]
    elif args["dataset_order"] == 3:
        dataset_order = ["['sgd_services_4']", "['sgd_hotels_3']", "['sgd_music_1']", "['sgd_flights_1']",
                         "['sgd_hotels_1']", "['sgd_hotels_4']", "['sgd_media_2']", "['sgd_flights_3']",
                         "['sgd_trains_1']", "['sgd_homes_1']", "['sgd_restaurants_1']", "['sgd_rentalcars_2']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_rentalcars_3']"]
    elif args["dataset_order"] == 4:
        dataset_order = ["['sgd_hotels_1']", "['sgd_media_2']", "['sgd_homes_1']", "['sgd_music_1']",
                         "['sgd_services_4']", "['sgd_restaurants_1']", "['sgd_flights_1']", "['sgd_hotels_4']",
                         "['sgd_services_3']", "['sgd_homes_2']", "['sgd_hotels_3']", "['sgd_trains_1']",
                         "['sgd_flights_3']", "['sgd_rentalcars_2']", "['sgd_rentalcars_3']"]
    elif args["dataset_order"] == 5:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_3']", "['sgd_homes_1']", "['sgd_flights_1']",
                         "['sgd_music_1']", "['sgd_services_3']", "['sgd_rentalcars_3']", "['sgd_media_2']",
                         "['sgd_restaurants_1']", "['sgd_hotels_1']", "['sgd_rentalcars_2']", "['sgd_hotels_4']",
                         "['sgd_hotels_3']", "['sgd_homes_2']", "['sgd_trains_1']"]
    elif args["dataset_order"] == 6:
        dataset_order = ["['sgd_restaurants_1']", "['sgd_services_3']", "['sgd_flights_1']", "['sgd_trains_1']",
                         "['sgd_hotels_1']", "['sgd_services_4']", "['sgd_hotels_3']", "['sgd_rentalcars_2']",
                         "['sgd_flights_3']", "['sgd_hotels_4']", "['sgd_homes_2']", "['sgd_homes_1']",
                         "['sgd_rentalcars_3']", "['sgd_media_2']", "['sgd_music_1']"]

    elif args["dataset_order"] == 99:
        # debug
        dataset_order = ["['sgd_hotels_4']", "['sgd_trains_1']"]

    elif args["dataset_order"] == 30:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order[-5:]
    elif args["dataset_order"] == 31:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order[-30:]
    elif args["dataset_order"] == 32:
        dataset_order = ["['sgd_events_3']", "['sgd_banks_2']", "['sgd_banks_1']", "['sgd_calendar_1']",
                         "['sgd_movies_3']", "['sgd_music_2']", "['sgd_services_2']", "['sgd_payment_1']",
                         "['sgd_media_1']", "['sgd_weather_1']", "['sgd_events_1']", "['sgd_flights_4']",
                         "['sgd_travel_1']", "['sgd_buses_2']", "['sgd_events_2']", "['sgd_alarm_1']",
                         "['sgd_buses_3']", "['sgd_services_1']", "['sgd_buses_1']", "['sgd_restaurants_2']",
                         "['sgd_hotels_2']", "['sgd_ridesharing_2']", "['sgd_rentalcars_1']", "['sgd_movies_1']",
                         "['sgd_ridesharing_1']", "['sgd_media_3']", "['sgd_music_3']", "['sgd_movies_2']",
                         "['sgd_flights_2']", "['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                         "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                         "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
        dataset_order = dataset_order
    else:
        raise
    
    model = T5ForPromptEncDecDST.from_pretrained('t5-small')      
    model.set_same_prompt_pos_emb() 
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    tokenizer.add_tokens(UNUSED_TOKENS)
    tokenizer.add_tokens(PROMPT_TOKENS)
    tokenizer.add_tokens(META_PROMPT_TOKENS)
    
    for name, x in model.named_parameters():
        if 'meta_prompt' not in name:
            x.requires_grad = False
            
    for name, x in model.named_parameters():
        if x.requires_grad:
            print(name)
    
    model.to(device)
    
    # prepare dataset
    train_file_path = args['train']
    dev_file_path = args['dev']
    test_file_path = args['test']
    with open(train_file_path, 'r') as f:
        train_file = json.load(f)
    with open(dev_file_path, 'r') as f:
        dev_file = json.load(f)
    with open(test_file_path, 'r') as f:
        test_file = json.load(f)

    new_train_file = []
    new_dev_file = []
    new_test_file = []

    test_dataset = dict()
    test_dataloader = dict()

    for services in dataset_order:
        new_train_file.extend([s for s in train_file if s['services'][0] == services[2:-2]])
        new_dev_file.extend([s for s in dev_file if s['services'][0] == services[2:-2]])
    train_dataset = DST_prompt_Dataset(args, tokenizer, new_train_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False)
    dev_dataset = DST_prompt_Dataset(args, tokenizer, new_dev_file)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args['batch_size'], shuffle=False)

    for services in dataset_order:
        test_dataset[services] = DST_prompt_Dataset(args, tokenizer, [s for s in test_file if s['services'][0] == services[2:-2]])
        test_dataloader[services] = DataLoader(test_dataset[services], batch_size=args['batch_size'], shuffle=False)

    # optimizer & scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['learning_rate'])

    num_training_steps = args['epochs'] * len(train_dataloader) - args['warmups']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args['warmups'],
                                                num_training_steps=args['epochs'] * len(train_dataloader))
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # train
    pbar = tqdm(range(args['epochs']), desc='Training')
    max_acc = -1000
    epoch = 0
    for i in pbar:
        ## train
        train_loss = 0
        train_cnt = 0
        for batch_idx, (_, input, input_mask, label, _, _) in enumerate(train_dataloader):

            input = input.to(device)
            label = label.to(device)
            input_mask = input_mask.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input.squeeze(1), 
                            decoder_input_ids=label.squeeze(1), 
                            attention_mask=input_mask.squeeze(1),
                            use_cache=False,
                            output_hidden_states=False,
                            return_dict=True,)
            lm_logits = outputs.logits
            loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label.view(-1))
            
            train_loss += loss
            train_cnt += 1
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"epoch": i + 1, "iter": batch_idx + 1, "train loss": loss.item()})

        # if epoch % 3 == 0:
        ## dev
        acc = 0
        total_acc = 0
        dev_cnt = 0
        dev_loss = 0
        with torch.no_grad():
            for _, (input_text, input, input_mask, label, label_text, slot_num) in enumerate(dev_dataloader):
                
                input = input.to(device)
                label = label.to(device)
                input_mask = input_mask.to(device)

                outputs = model.generate(input.squeeze(1), 
                                         attention_mask=input_mask, 
                                         use_cache=False,
                                         return_dict_in_generate=True,
                                         max_length=100)
                outputs_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                total_acc += cal_acc(outputs_texts, label_text, slot_num)
                
                lm_logits = outputs.logits
                dev_loss += loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label.view(-1))
                dev_cnt += 1

                # for i in range(len(label_text)):
                #     outputs_text = tokenizer.decode(outputs[i], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                #     if label_text[i] == outputs_text:
                #         total_acc += 1

        acc = total_acc / len(dev_dataset)
        print("epoch:", epoch + 1, "dev_acc:", acc)
        wandb.log({'dev_acc': acc, 'dev_loss': dev_loss/dev_cnt, 'loss': train_loss/train_cnt, 'learning_rate': get_lr(optimizer)})
        if acc >= max_acc:
            best_prompt_embedding = model.shared
            max_acc = acc
        # else:
        #     wandb.log({'loss': train_loss/train_cnt, 'learning_rate': get_lr(optimizer)})
        epoch += 1
    
    ## test
    acc = 0
    total_acc = 0
    # model.set_input_embeddings(best_prompt_embedding)
    avg_jga = 0
    for services in dataset_order:
        acc = 0
        total_acc = 0
        with torch.no_grad():
            for _, (input_text, input, input_mask, label, label_text, slot_num) in enumerate(test_dataloader[services]):
                
                input = input.to(device)
                label = label.to(device)
                input_mask = input_mask.to(device)
                
                outputs = model.generate(input.squeeze(1), 
                                         attention_mask=input_mask, 
                                         use_cache=False,
                                         return_dict_in_generate=True,
                                         max_length=100)
                outputs_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                total_acc += cal_acc(outputs_texts, label_text, slot_num)
                
                # for i in range(len(label_text)):
                #     outputs_text = tokenizer.decode(outputs[i], skip_special_tokens=False)
                #     if label_text[i] == outputs_text:
                #         total_acc += 1
        acc = total_acc / len(test_dataset[services])
        avg_jga += acc
        print(services[2:-2], "test_acc:", acc)
    print("avg_jga:", avg_jga/len(dataset_order))
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/sgd_train.json')
    parser.add_argument('--dev', type=str, default='data/sgd_valid.json')
    parser.add_argument('--test', type=str, default='data/sgd_test.json')
    parser.add_argument('--schema', type=str, default='data/all_schema.json')
    parser.add_argument('--task', type=str, default='MULTI')
    parser.add_argument('--warmups', type=int, default=0, help='warmups')
    parser.add_argument('--epochs', type=int, default=0, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--gpu-id', type=str, default='1')

    parser.add_argument('--max_history', type=int, default=100, help='max_history')
    parser.add_argument('--max_length', type=int, default=1024, help='max_input_length')
    parser.add_argument('--prompt_length', type=int, default=100, help='prompt_length')
    parser.add_argument('--data_type', type=str, default='all_history_multi_prompt')
    parser.add_argument('--num_prompt_tokens', type=int, default=200)
    parser.add_argument('--num_meta_prompt_tokens', type=int, default=200)
    parser.add_argument('--dataset_order', type=int, default=1)


    args = parser.parse_args()
    args = args.__dict__

    main(args)