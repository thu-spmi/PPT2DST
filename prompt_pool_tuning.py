import os
import json
import argparse
import numpy as np
import torch
import re
import random
import matplotlib.pyplot as plt
from load_data import DST_prompt_pool_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from heatmap_pic import draw_heatmap

# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertTokenizer

from tqdm import tqdm
import wandb

from model import T5ForPromptPool
from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer
from prompt_files.prompts_config import UNUSED_TOKENS, PROMPT_TOKENS, META_PROMPT_TOKENS

wandb.init(project="prompt_pool_tuning_w_replay_ch_multi&cl")

def draw_dis(args, services_id, epoch, fig_type, select_prompt):
    
    folder = 'frequency_res/replay_' + args['model_name'] + '_' + args['dis_method'] + '_mem' + str(args['M']) + '_'+ str(args['pool_size'])+'/' + str(services_id)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if fig_type in ['train', 'dev']:
        pic_dir = folder+'/'+fig_type+'_'+str(epoch)+'.png'
    else: pic_dir = folder+'/'+fig_type+'.png'
    
    dis_data = np.zeros(args['pool_size'])
    for prompts in select_prompt:
        for p in prompts:
            # num = int(p[0][-2:])
            dis_data[p] += 1
    dis_data = dis_data / len(select_prompt)
    
    plt.figure(dpi=300,figsize=(30,8))
    plt.xticks(fontsize=10)
    plt.bar(range(int(len(dis_data))), dis_data, tick_label=range(int(len(dis_data))))  
    
    f = plt.gcf()
    f.savefig(pic_dir)
    f.clear()  
    
    plt.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cal_acc(pre, label, slot_num):

    dst_predictions = [re.sub('\<(extra_id__)(.*?)\>', '', s) for s in pre]
    dst_predictions = [_.strip() for _ in dst_predictions]
    acc = 0
    for i, pred_str in enumerate(dst_predictions):
        new_pre = ''
        for j in range(slot_num[i]+1):
            left_token = '<extra_id_{}>'.format(j)
            right_token = '<extra_id_{}>'.format(j + 1)
            if right_token not in pred_str:
                right_token = '</s>'
            try:
                value = pred_str.split(left_token)[1].split(right_token)[0].strip()
            except:
                value = '无'
            new_pre += '<extra_id_{}>'.format(j) + value
        if new_pre == label[i]:
            acc += 1
    return acc

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        print ('CUDA is available')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
                            
        
def main(args):
    
    device = torch.device("cuda:{}".format(args['gpu_id']) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # seed_all(args["seed"])
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
    elif args["dataset_order"] == 7:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                         "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_media_2']", "['sgd_hotels_1']"]
    elif args["dataset_order"] == 8:
        dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_media_2']"]
    elif args["dataset_order"] == 10:
        dataset_order = ["搬家", "会议", "娱乐", "挪车", "装修", "约车",
                         "商务", "电商", "产品", "卖车", "广告", "售楼推销", 
                         "回访", "催缴欠款", "课程推销", "通知"]
    elif args["dataset_order"] == 11:
        dataset_order = ["多任务", "售楼推销", "回访", "催缴欠款", "课程推销", "通知"]
        multi_task =  ["搬家", "会议", "娱乐", "挪车", "装修", "约车", "商务", "电商", "产品", "卖车", "广告" ]

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
    
    tokenizer = T5Tokenizer.from_pretrained(args['model_name'])
    # tokenizer = BertTokenizer.from_pretrained(args['model_name'])
    tokenizer.add_tokens(UNUSED_TOKENS)
    tokenizer.add_tokens(PROMPT_TOKENS)
    tokenizer.add_tokens(META_PROMPT_TOKENS)
    
    model = T5ForPromptPool(args, tokenizer, dataset_order)
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

    train_dataset = dict()
    train_dataloader = dict()
    dev_dataset = dict()
    dev_dataloader = dict()
    test_dataset = dict()
    test_dataloader = dict()
    
    memories = []
    fix_size_memeories = []
    memory_per_task = []
    if args['dataset'] == 'sgd':
        for id, services in enumerate(dataset_order):
            
            train_samples = [s for s in train_file if s['services'][0] == services[2:-2]]
            for s_id, sample in enumerate(train_samples):
                train_samples[s_id]['task_id'] = id
            train_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, train_samples, data_id=id)
            train_dataloader[services] = DataLoader(train_dataset[services], batch_size=args['batch_size'], shuffle=True, collate_fn=train_dataset[services].collate_fn)
            
            fix_size_memeories = []
            samples = train_dataset[services].sample(args['M'])
            memory_per_task.append(samples)
            if id > 0 and args['M'] > 0:
                memory_num_per_task = int(args['M'] / id)
                for t in range(id):
                    fix_size_memeories.extend(memory_per_task[t][:memory_num_per_task+1])
                temp = args['M'] - len(fix_size_memeories)
                fix_size_memeories.extend(memory_per_task[id-1][memory_num_per_task+1:memory_num_per_task+temp+2])
            if args['memory_type'] == 'fix_size':
                train_dataset[services].add_sample(fix_size_memeories)
            else:
                train_dataset[services].add_sample(memories)
            memories.extend(samples)
            
            dev_samples = [s for s in dev_file if s['services'][0] == services[2:-2]]
            for s_id, sample in enumerate(dev_samples):
                dev_samples[s_id]['task_id'] = id
            dev_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, dev_samples, data_id=id)
            dev_dataloader[services] = DataLoader(dev_dataset[services], batch_size=args['batch_size'], shuffle=False, collate_fn=dev_dataset[services].collate_fn)
            
            test_samples = [s for s in test_file if s['services'][0] == services[2:-2]]
            for s_id, sample in enumerate(test_samples):
                test_samples[s_id]['task_id'] = id
            test_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, test_samples, data_id=id)
            test_dataloader[services] = DataLoader(test_dataset[services], batch_size=args['batch_size'], shuffle=False, collate_fn=test_dataset[services].collate_fn)
    elif args['dataset'] == 'xiaomishu':
        for id, services in enumerate(dataset_order):
            if services == '多任务':
                train_samples = []
                dev_samples = []
                for m_id, m_services in enumerate(multi_task):
                    m_train_samples = [s for s in train_file if s['log'][0]['domain'] == m_services]
                    for s_id, sample in enumerate(m_train_samples):
                        m_train_samples[s_id]['task_id'] = m_id
                    train_samples.extend(m_train_samples)
                    
                    m_dev_samples = [s for s in dev_file if s['log'][0]['domain'] == m_services]
                    for s_id, sample in enumerate(m_dev_samples):
                        m_dev_samples[s_id]['task_id'] = m_id
                    dev_samples.extend(m_dev_samples)
                    
                    m_test_samples = [s for s in test_file if s['log'][0]['domain'] == m_services]
                    for s_id, sample in enumerate(m_test_samples):
                        m_test_samples[s_id]['task_id'] = m_id
                    test_dataset[m_services] = DST_prompt_pool_Dataset(args, tokenizer, m_test_samples, data_id=id)
                    test_dataloader[m_services] = DataLoader(test_dataset[m_services], batch_size=args['batch_size'], shuffle=False, collate_fn=test_dataset[m_services].collate_fn)
                    
                train_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, train_samples, data_id=id)
                train_dataloader[services] = DataLoader(train_dataset[services], batch_size=args['batch_size'], shuffle=True, collate_fn=train_dataset[services].collate_fn)
                dev_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, dev_samples, data_id=id)
                dev_dataloader[services] = DataLoader(dev_dataset[services], batch_size=args['batch_size'], shuffle=True, collate_fn=dev_dataset[services].collate_fn)
            else: 
                train_samples = [s for s in train_file if s['log'][0]['domain'] == services]
                for s_id, sample in enumerate(train_samples):
                    train_samples[s_id]['task_id'] = id + len(multi_task) - 1
                train_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, train_samples, data_id=id)
                train_dataloader[services] = DataLoader(train_dataset[services], batch_size=args['batch_size'], shuffle=True, collate_fn=train_dataset[services].collate_fn)
                
                # fix_size_memeories = []
                # samples = train_dataset[services].sample(args['M'])
                # memory_per_task.append(samples)
                # if id > 0 and args['M'] > 0:
                #     memory_num_per_task = int(args['M'] / id)
                #     for t in range(id):
                #         fix_size_memeories.extend(memory_per_task[t][:memory_num_per_task+1])
                #     temp = args['M'] - len(fix_size_memeories)
                #     fix_size_memeories.extend(memory_per_task[id-1][memory_num_per_task+1:memory_num_per_task+temp+2])
                # if args['memory_type'] == 'fix_size':
                #     train_dataset[services].add_sample(fix_size_memeories)
                # else:
                #     train_dataset[services].add_sample(memories)
                # memories.extend(samples)
                
                dev_samples = [s for s in dev_file if s['log'][0]['domain'] == services]
                for s_id, sample in enumerate(dev_samples):
                    dev_samples[s_id]['task_id'] = id
                dev_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, dev_samples, data_id=id)
                dev_dataloader[services] = DataLoader(dev_dataset[services], batch_size=args['batch_size'], shuffle=False, collate_fn=dev_dataset[services].collate_fn)
                
                test_samples = [s for s in test_file if s['log'][0]['domain'] == services]
                for s_id, sample in enumerate(test_samples):
                    test_samples[s_id]['task_id'] = id
                test_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, test_samples, data_id=id)
                test_dataloader[services] = DataLoader(test_dataset[services], batch_size=args['batch_size'], shuffle=False, collate_fn=test_dataset[services].collate_fn)
    
    # train
    # torch.autograd.set_detect_anomaly(True)
    model.initialize_penalty()
    # draw_heatmap(model.pool_keys, 'init', -1, dataset_order, args["pool_size"], args)
    best_acc = torch.zeros(len(dataset_order))
    forgetting = torch.zeros((len(dataset_order), len(dataset_order)))
    
    save_path = args["save_path"] + '_' + str(args["pool_size"]) + '.pt'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    # sorted_idx = torch.arange(service_id * args['top_n'], (service_id+1) * args['top_n'])
    # sorted_idx = sorted_idx.repeat(args['batch_size'], 1)
    sorted_idx = None
    
    memory_data = []
    
    for service_id, services in  enumerate(dataset_order):
        # optimizer & scheduler
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args['learning_rate'], eps=args['adamw_eps'], weight_decay=args['weight_decay'])

        num_training_steps = args['epochs'] * len(train_dataloader[services])
        # if args['memory_replay']:
        #     for i in range(service_id):
        #         num_training_steps += args['epochs'] * len(memory_dataloader[dataset_order[i]])
        num_warmup_steps = int(num_training_steps*args['warmups'])
        num_training_steps = num_training_steps - num_warmup_steps
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        
        print(services[2:-2])
        pbar = tqdm(range(args['epochs']), desc='Training')
        max_acc = -1000
        epoch = 0
        
        # update hidden_mean
        # model.update_mean(service_id, dataset_order)
    
        for i in pbar:
            
            train_prompt_dis = []
            dev_prompt_dis = []
            
            ## train
            train_loss = 0
            train_model_loss = 0
            train_dis_loss = 0
            train_cnt = 0
            
            task_sorted_idx_init = torch.arange(service_id * args['top_n'], (service_id+1) * args['top_n'])
            # sorted_idx = sorted_idx.repeat(args['batch_size'], 1)
            # service_id_label = torch.ones(args['batch_size']) * service_id
            
            for batch_idx, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, _, input_task_id, input_sorted_idx) in enumerate(train_dataloader[services]):

                input = input.to(device)
                input_mask = input_mask.to(device)
                label = label.to(device)
                dst_decoder_input_ids = dst_decoder_input_ids.to(device)
                
                optimizer.zero_grad()
                
                if args['penalty']:
                    sorted_idx_key = input_sorted_idx
                    sorted_idx = task_sorted_idx_init.repeat(len(input_text), 1)
                else:
                    sorted_idx = None
                loss, m_loss, s_loss, train_selected_ptompt = model(input_text, input, input_mask, dst_decoder_input_ids, label, device, services, service_id, input_task_id, input_sorted_idx, sorted_idx, sorted_idx_key)
                train_prompt_dis.extend(train_selected_ptompt)
                    
                train_loss += loss.item()
                train_model_loss += m_loss
                train_dis_loss += s_loss
                train_cnt += 1
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"epoch": i + 1, "iter": batch_idx + 1, "model_loss": m_loss, "similarity_loss": s_loss})

            ## dev
            acc = 0
            total_acc = 0
            selected_acc = 0
            total_selected_acc = 0
            
            dev_cnt = 0
            dev_loss = 0
            with torch.no_grad():
                for _, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, slot_num, input_task_id, input_sorted_idx) in enumerate(dev_dataloader[services]):
                    
                    input = input.to(device)
                    label = label.to(device)
                    input_mask = input_mask.to(device)
                    dst_decoder_input_ids = dst_decoder_input_ids.to(device)
                    dev_selected_prompt_label = [torch.arange(x*args['top_n'], (x+1)*args['top_n']) for x in torch.ones(input_task_id.shape)*service_id]
                    
                    if args['prompt_tuning']:
                        sorted_idx_key = [torch.range(x * args['top_n'], (x+1) * args['top_n']) for x in input_sorted_idx]
                        sorted_idx_key = torch.Tensor(sorted_idx_key)
                        sorted_idx = task_sorted_idx_init.repeat(len(input_text), 1)
                    else:
                        sorted_idx = None
                        sorted_idx_key = None
                    
                    outputs, dev_selected_prompt = model.generate(input_text, input, input_mask, device, services, sorted_idx)
                    outputs_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                    total_acc += cal_acc(outputs_texts, label_text, slot_num)
                    dev_selected_acc = [torch.equal(dev_selected_prompt_label[j], torch.tensor(dev_selected_prompt[j], dtype=torch.int64)) for j in range(len(dev_selected_prompt))]
                    total_selected_acc += dev_selected_acc.count(True)
                    
                    dev_prompt_dis.extend(dev_selected_prompt)
                    dev_loss, dev_model_loss, dev_dis_loss, _ = model(input_text, input, input_mask, dst_decoder_input_ids, label, device, services, service_id, input_task_id, input_sorted_idx, sorted_idx)
                    dev_cnt += 1
            
            acc = total_acc / len(dev_dataset[services])
            selected_acc = total_selected_acc / len(dev_dataset[services])
            
            print("epoch:", epoch + 1, "dev_acc:", round(acc, 3), "dev_selected_acc", round(selected_acc, 3))
            wandb.log({'dev_acc': acc, 'dev_selected_acc': selected_acc, 'dev_loss': dev_loss/dev_cnt, 'dev_model_loss': dev_model_loss/dev_cnt, 'dev_similarity_loss': dev_dis_loss/dev_cnt, 'train_loss': train_loss/train_cnt, 'train_model_loss': train_model_loss/train_cnt, 'train_similarity_loss': train_dis_loss/train_cnt, 'learning_rate': get_lr(optimizer)})
            if acc > max_acc:
                torch.save(model.state_dict(), save_path)
                max_acc = acc
            epoch += 1
            
        draw_dis(args, service_id, epoch, 'train', train_prompt_dis)
        draw_dis(args, service_id, epoch, 'dev', dev_prompt_dis)  
          
        # model.update_penalty()
        model.load_state_dict(torch.load(save_path))
        # draw_heatmap(model.pool_keys, services[2:-2], service_id, dataset_order, args["pool_size"], args)
        # test_dataset_order = multi_task
        # test_dataset_order.extend(dataset_order[1:])
        # best_acc, forgetting = test(model, tokenizer, len(multi_task)+service_id-1, test_dataset_order, test_dataloader, test_dataset, device, best_acc, forgetting)
    
    ## test
    # model.load_state_dict(torch.load(args["save_path"]))
    
    # print(best_acc)
    # print(forgetting)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    acc = 0
    total_acc = 0
    avg_jga = 0
    dataset_order_new = multi_task.copy()
    dataset_order_new.extend(dataset_order[1:])
    for service_id_new, services in enumerate(dataset_order_new):
        acc = 0
        total_acc = 0
        selected_acc = 0
        total_selected_acc = 0
        test_prompt_dis = []
        if services in multi_task:
            service_id = 0
        else: service_id = service_id_new - len(multi_task) + 1
        task_sorted_idx_init = torch.arange(service_id * args['top_n'], (service_id+1) * args['top_n'])
        
        with torch.no_grad():
            for _, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, slot_num, input_task_id, input_sorted_idx) in enumerate(test_dataloader[services]):
                
                input = input.to(device)
                label = label.to(device)
                input_mask = input_mask.to(device)
                dst_decoder_input_ids = dst_decoder_input_ids.to(device)
                test_selected_prompt_label = [torch.arange(x*args['top_n'], (x+1)*args['top_n']) for x in torch.ones(input_task_id.shape)*service_id]
                
                if args['prompt_tuning']:
                    sorted_idx = task_sorted_idx_init.repeat(len(input_text), 1)
                else:
                    sorted_idx = None
                
                outputs, test_selected_prompt = model.generate(input_text, input, input_mask, device, services, sorted_idx)
                outputs_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                total_acc += cal_acc(outputs_texts, label_text, slot_num)
                test_prompt_dis.extend(test_selected_prompt)
                test_selected_acc = [torch.equal(test_selected_prompt_label[j], torch.tensor(test_selected_prompt[j], dtype=torch.int64)) for j in range(len(test_selected_prompt))]
                total_selected_acc += test_selected_acc.count(True)
                
        acc = total_acc / len(test_dataset[services])
        selected_acc = total_selected_acc / len(test_dataset[services])
        avg_jga += acc
        print(services, "test_acc:", round(acc, 3), "test_selected_acc", round(selected_acc, 3))
        # draw_dis(args, service_id, 0, 'test', test_prompt_dis)
    print("avg_jga:", round(avg_jga/len(dataset_order), 3))
    
def test(model, tokenizer, task_id, dataset_order, test_dataloader, test_dataset, device, best_acc, forgetting):
    model.eval()
    acc = 0
    total_acc = 0
    avg_jga = 0
    for idx, services in enumerate(dataset_order[:task_id + 1]):
        acc = 0
        total_acc = 0
        test_prompt_dis = []
        task_sorted_idx_init = torch.arange(idx * args['top_n'], (idx+1) * args['top_n'])
        
        with torch.no_grad():
            for _, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, slot_num, input_task_id, input_sorted_idx) in enumerate(test_dataloader[services]):
                
                input = input.to(device)
                label = label.to(device)
                input_mask = input_mask.to(device)
                dst_decoder_input_ids = dst_decoder_input_ids.to(device)
                
                if args['prompt_tuning']:
                    sorted_idx = task_sorted_idx_init.repeat(len(input_text), 1)
                else:
                    sorted_idx = None
                
                outputs, test_selected_prompt = model.generate(input_text, input, input_mask, device, services, sorted_idx)
                outputs_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                total_acc += cal_acc(outputs_texts, label_text, slot_num)
                test_prompt_dis.extend(test_selected_prompt)
                
        acc = total_acc / len(test_dataset[services])
        if idx < task_id:
            forgetting[task_id, idx] = best_acc[idx] - acc
        best_acc[idx] = max(best_acc[idx], acc)
        avg_jga += acc
        
    model.train()
    return best_acc, forgetting

def train_memory(model, tokenizer, task_id, dataset_order, memory_dataloader, memory_dataset, device, optimizer, scheduler, pbar):
    
    for service_id, services in  enumerate(dataset_order[:task_id]):
        
        sorted_idx = torch.arange(service_id * args['top_n'], (service_id+1) * args['top_n'])
        sorted_idx = sorted_idx.repeat(args['batch_size'], 1)
        service_id_label = torch.ones(args['batch_size']) * service_id
        
        for batch_idx, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, _) in enumerate(memory_dataloader[services]):

            input = input.to(device)
            input_mask = input_mask.to(device)
            label = label.to(device)
            dst_decoder_input_ids = dst_decoder_input_ids.to(device)
            
            optimizer.zero_grad()
            
            loss, m_loss, s_loss, train_selected_ptompt = model(input_text, input, input_mask, dst_decoder_input_ids, label, device, services, sorted_idx)
                
            train_loss += loss.item()
            train_model_loss += m_loss
            train_dis_loss += s_loss
            train_cnt += 1
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"epoch": i + 1, "iter": batch_idx + 1, "model_loss": m_loss, "similarity_loss": s_loss})
    
    
    pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/xiaomishu_train.json')
    parser.add_argument('--dev', type=str, default='data/xiaomishu_eval.json')
    parser.add_argument('--test', type=str, default='data/xiaomishu_test.json')
    parser.add_argument('--schema', type=str, default='data/xiaomishu_schema.json')
    parser.add_argument('--save_path', type=str, default='/mnt/workspace/zhouyuan/L2P4DST/prompt_pool_model_ch_multi&cl_5')
    parser.add_argument('--task', type=str, default='MULTI')
    parser.add_argument('--select_method', type=str, default='bert-encoder-ch')
    parser.add_argument('--warmups', type=int, default=0, help='warmups')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=0.25, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--adamw_eps', type=float, default=1e-8, help='adamw_epsilon')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--gpu_id', type=str, default='9')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--max_history', type=int, default=100, help='max_history')
    parser.add_argument('--max_length', type=int, default=1024, help='max_input_length')
    parser.add_argument('--prompt_length', type=int, default=10, help='prompt_length')
    parser.add_argument('--data_type', type=str, default='all_history_prompt_tuning_bce')
    parser.add_argument('--num_prompt_tokens', type=int, default=200)
    parser.add_argument('--num_meta_prompt_tokens', type=int, default=200)
    parser.add_argument('--dataset_order', type=int, default=11)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--pool_size', type=int, default=160)
    parser.add_argument('--lambda', type=float, default=0.001)
    
    parser.add_argument('--penalty', action='store_true')
    parser.add_argument('--M', type=int, default=0)
    parser.add_argument('--memory_replay', type=bool, default=True)
    parser.add_argument('--dis_method', type=str, default='euclidean') # 'cosin', 'bce', 'euclidean', 'bce-bond', 'euclidean-bond', 'gau-bond'
    
    parser.add_argument('--model_name', type=str, default='google/mt5-small')
    parser.add_argument('--dataset', type=str, default='xiaomishu')
    parser.add_argument('--prompt_tuning', type=bool, default=False)
    parser.add_argument('--memory_type', type=str, default='incre_size')

    args = parser.parse_args()
    args = args.__dict__

    main(args)