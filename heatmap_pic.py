import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

import argparse
import os

from transformers import AutoTokenizer, BertModel, AutoModel

import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from load_data import DST_prompt_pool_Dataset
from torch.utils.data import DataLoader

from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cal_mean(args):

    device = torch.device("cuda:{}".format(args['gpu_id']) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # seed_all(args["seed"])
    # wandb.config.update(args)
    
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
    elif args["dataset_order"] == 10:
        dataset_order = ["搬家", "会议", "娱乐", "挪车", "装修", "约车",
                         "商务", "电商", "产品", "卖车", "广告", "售楼推销", 
                         "回访", "催缴欠款", "课程推销", "通知"]
    
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
    
    if args['select_method'] == 't5':
        model = T5ForPromptEncDecDST.from_pretrained('t5-base')
    elif args['select_method'] == 'bert':
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif args['select_method'] == 'bert-encoder':
        bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    elif args['select_method'] == 'bert-encoder-ch':
        bert_tokenizer = AutoTokenizer.from_pretrained('uer/sbert-base-chinese-nli')
        model = AutoModel.from_pretrained('uer/sbert-base-chinese-nli')
        
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
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
    
    if args['dataset'] == 'sgd':
        for id, services in enumerate(dataset_order):
            train_samples = [s for s in train_file if s['services'][0] == services[2:-2]]
            for s_id, sample in enumerate(train_samples):
                train_samples[s_id]['task_id'] = id
            train_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, train_samples, data_id=id)
            train_dataloader[services] = DataLoader(train_dataset[services], batch_size=args['batch_size'], shuffle=True, collate_fn=train_dataset[services].collate_fn)
            
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
            train_samples = [s for s in train_file if s['log'][0]['domain'] == services]
            for s_id, sample in enumerate(train_samples):
                train_samples[s_id]['task_id'] = id
            train_dataset[services] = DST_prompt_pool_Dataset(args, tokenizer, train_samples, data_id=id)
            train_dataloader[services] = DataLoader(train_dataset[services], batch_size=args['batch_size'], shuffle=True, collate_fn=train_dataset[services].collate_fn)
            
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
    
    input_hiddens = np.zeros((0, 768))
    input_services = []
    services_mean = dict()
    all_services_mean = dict()
    test_all_services_mean = dict()
    all_input_hiddens = np.zeros((0, 768))
    
    for services in tqdm(dataset_order):
        s_input_hiddens = np.zeros((0, 768))
        service_mean = np.zeros((1, 768))
        for batch_idx, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, _, input_task_id, input_sorted_idx) in enumerate(train_dataloader[services]):
            
            if args['select_method'] == 't5':
                input_hidden = model(input_ids=input,  
                                    attention_mask=input_mask,
                                    use_cache=False,
                                    output_hidden_states=False,
                                    return_dict=True,
                                    only_encoder=True).encoder_last_hidden_state[:, :]
                input_hidden = torch.mean(input_hidden, dim=1)
            
            elif args['select_method'] == 'bert':
                bert_input = bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
                input_hidden = model(**bert_input).last_hidden_state
                input_hidden = input_hidden[:,0,:]
                
            elif args['select_method'] == 'bert-encoder':
                encoded_input = bert_tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
                
            elif args['select_method'] == 'bert-encoder-ch':
                encoded_input = bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt')
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
            
            
            input_hidden = input_hidden.detach().numpy()
            input_hiddens = np.append(input_hiddens, input_hidden, axis=0)
            all_input_hiddens = np.append(all_input_hiddens, input_hidden, axis=0)
            service_mean += np.sum(input_hidden, axis=0)
            s_input_hiddens = np.append(s_input_hiddens, input_hidden, axis=0)
            
            input_service = [services] * len(input_text)
            input_services.extend(input_service)
        
        # test_s_input_hiddens = np.zeros((0, 768))   
        # for batch_idx, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, _, input_task_id, input_sorted_idx) in enumerate(test_dataloader[services]):
            
        #     if args['select_method'] == 't5':
        #         input_hidden = model(input_ids=input,  
        #                             attention_mask=input_mask,
        #                             use_cache=False,
        #                             output_hidden_states=False,
        #                             return_dict=True,
        #                             only_encoder=True).encoder_last_hidden_state[:, :]
        #         input_hidden = torch.mean(input_hidden, dim=1)
            
        #     elif args['select_method'] == 'bert':
        #         bert_input = bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
        #         input_hidden = model(**bert_input).last_hidden_state
        #         input_hidden = input_hidden[:,0,:]
                
        #     elif args['select_method'] == 'bert-encoder':
        #         encoded_input = bert_tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        #         model_output = model(**encoded_input)
        #         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        #         input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
                
        #     elif args['select_method'] == 'bert-encoder-ch':
        #         encoded_input = bert_tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        #         model_output = model(**encoded_input)
        #         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        #         input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # input_hidden = input_hidden.detach().numpy()
            # input_hiddens = np.append(input_hiddens, input_hidden, axis=0)
            # all_input_hiddens = np.append(all_input_hiddens, input_hidden, axis=0)
            # service_mean += np.sum(input_hidden, axis=0)
            # test_s_input_hiddens = np.append(test_s_input_hiddens, input_hidden, axis=0)
            
            # input_service = [services] * len(input_text)
            # input_services.extend(input_service)
            
        print(services[2:-2], s_input_hiddens.mean(), s_input_hiddens.var())
        all_services_mean[services] = s_input_hiddens
        # test_all_services_mean[services] = test_s_input_hiddens
        service_mean = service_mean / len(train_dataset[services])
        services_mean[services] = service_mean
    
    all_service_mean = all_input_hiddens.mean(axis=0)
    
    np.save('services_encoding_mean_bert-encoder-ch.npy', services_mean, allow_pickle=True)
    np.save('services_mean_bert-encoder-ch.npy', all_service_mean)
    
    # np.save('train_all_services_encoding_mean_bert-encoder.npy', all_services_mean, allow_pickle=True)
    # np.save('test_all_services_encoding_mean_bert-encoder.npy', test_all_services_mean, allow_pickle=True)
    
    return services_mean

def cal_Euclidean_Distance(key, mean):
    
    key = torch.from_numpy(key)
    mean = torch.from_numpy(mean).to(key.device)
    
    distance = torch.sqrt(torch.sum((key-mean) ** 2, dim=1, keepdim=True))
    similarity = 1 / (distance + 1)
    
    return similarity
    

def cal_similarity(key, mean, epsilon=1e-12):
    
    key = torch.from_numpy(key)
    mean = torch.from_numpy(mean).to(key.device)
    # mean = mean.to(torch.float32)
    hidden_states_square_sum = torch.sum(mean ** 2, dim=1, keepdim=True) # size: (1, 512)
    prompt_key_square_sum = torch.sum(key ** 2, dim=1, keepdim=True) # size: (100, 512)
    
    norm_similarity = torch.matmul(hidden_states_square_sum, prompt_key_square_sum.t()) # size: (1, 100)
    similarity = torch.matmul(mean, key.t()) * torch.rsqrt(torch.maximum(norm_similarity, torch.tensor(epsilon, device=key.device)))
    
    return similarity
    
def draw_heatmap(pool_keys, train_service, service_id, dataset_order, pool_size, args):
    
    service_mean_path = 'services_encoding_mean_' + args['select_method'] + '.npy'
    mean_path = 'services_mean_' + args['select_method'] + '.npy'
    
    services_mean = np.load(service_mean_path, allow_pickle=True).item()
    mean = np.load(mean_path)
    services_sim = np.zeros((len(dataset_order), len(pool_keys)))
    for i, key in enumerate(dataset_order): 
        # # cosin_sim
        # service_sim = cal_similarity(pool_keys, services_mean[key] - mean).cpu().detach().numpy()
        
        # # bce_loss
        # temp = services_mean[key] - mean
        # temp = torch.from_numpy(temp).to(pool_keys.device).to(pool_keys.dtype)
        # service_sim = torch.matmul(temp, pool_keys.t()).cpu().detach().numpy()
        # services_sim[i] = service_sim
        
        # euclidean_loss
        temp = torch.from_numpy(services_mean[key]).to(pool_keys.device).to(pool_keys.dtype)
        service_sim = torch.exp(-torch.sum((temp - pool_keys) ** 2, dim=-1, keepdim=True).t()).cpu().detach().numpy()
        services_sim[i] = service_sim
        
    data=pd.DataFrame(services_sim)
    fig=plt.figure(figsize=(30,5),dpi=300) 
    plot=sns.heatmap(data, linewidths = 0.05, vmax=1, vmin=0)
    
    folder = 'heatmaps/replay_' + args['model_name'] + '_' + args['dis_method'] + '_mem' + str(args['M']) + '_'+ str(pool_size)
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_file = folder + '/zero_heatmap_' + str(service_id + 1) + '.png'
    plot.get_figure().savefig(save_file)
    plt.close()

def draw_services_heatmap():
    
    services_mean = np.load('services_encoding_mean_bert-encoder.npy', allow_pickle=True).item()
    services_sim = np.zeros((15, 15))
    mean = np.zeros((1,384))
    for key in services_mean.keys():
        mean += services_mean[key]
    mean = mean / 15
    # np.save('services_mean.npy', mean)
    for i, key in enumerate(services_mean.keys()): 
        for j, key1 in enumerate(services_mean.keys()):
            service_sim = cal_similarity(services_mean[key] - mean, services_mean[key1] - mean).cpu().detach().numpy()
            services_sim[i,j] = service_sim
        
    data=pd.DataFrame(services_sim)
    fig=plt.figure(figsize=(30,30),dpi=300) 
    plot=sns.heatmap(data, linewidths = 0.05, vmax=1, vmin=-1)
    
    save_file = 'bert-encoder_services_sim_heatmap.png'
    plot.get_figure().savefig(save_file)
    plt.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/xiaomishu_train.json')
    parser.add_argument('--dev', type=str, default='data/xiaomishu_eval.json')
    parser.add_argument('--test', type=str, default='data/xiaomishu_test.json')
    parser.add_argument('--schema', type=str, default='data/xiaomishu_schema.json')
    parser.add_argument('--save_path', type=str, default='/mnt/workspace/zhouyuan/L2P4DST/best_prompt_pool_model_lr3e-2.pt')
    parser.add_argument('--task', type=str, default='MULTI')
    parser.add_argument('--select_method', type=str, default='bert-encoder-ch')
    parser.add_argument('--warmups', type=int, default=0, help='warmups')
    parser.add_argument('--epochs', type=int, default=0, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--adamw_eps', type=float, default=1e-8, help='adamw_epsilon')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--gpu-id', type=str, default='3')
    parser.add_argument('--seed', type=int, default=1)
    

    parser.add_argument('--max_history', type=int, default=100, help='max_history')
    parser.add_argument('--max_length', type=int, default=1024, help='max_input_length')
    parser.add_argument('--prompt_length', type=int, default=10, help='prompt_length')
    parser.add_argument('--data_type', type=str, default='all_history_prompt_tuning')
    parser.add_argument('--num_prompt_tokens', type=int, default=200)
    parser.add_argument('--num_meta_prompt_tokens', type=int, default=200)
    parser.add_argument('--dataset_order', type=int, default=10)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--lambda', type=float, default=0.5)
    
    parser.add_argument('--multi_num', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='xiaomishu')


    args = parser.parse_args()
    args = args.__dict__

    # draw_services_heatmap()
    cal_mean(args)