import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

import argparse

from transformers import AutoTokenizer, BertModel

import json
from tqdm import tqdm
import numpy as np
import torch
from load_data import DST_prompt_pool_Dataset
from torch.utils.data import DataLoader

from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main(args):

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
        model = T5ForPromptEncDecDST.from_pretrained('t5-small')
    elif args['select_method'] == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased')
        
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    model.eval()  
      
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
    
    input_hiddens = np.zeros((0,512))
    input_services = []
    
    for services in tqdm(dataset_order):
        s_input_hiddens = np.zeros((0,512))
        for batch_idx, (input_text, input, input_mask, dst_decoder_input_ids, label, label_text, _, input_task_id, input_sorted_idx) in enumerate(train_dataloader[services]):
            
            if args['select_method'] == 't5':
                input_hidden = model(input_ids=input,  
                                    attention_mask=input_mask,
                                    use_cache=False,
                                    output_hidden_states=False,
                                    return_dict=True,
                                    only_encoder=True).encoder_last_hidden_state[:, :]
                input_hidden = torch.mean(input_hidden, dim=1)
                # input_hidden = mean_pooling(input_hidden, input_mask)
            
            elif args['select_method'] == 'bert':
                bert_input = bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
                input_hidden = model(**bert_input).last_hidden_state
                input_hidden = input_hidden[:,0,:]
            
            
            input_hidden = input_hidden.detach().numpy()
            input_hiddens = np.append(input_hiddens, input_hidden, axis=0)
            s_input_hiddens = np.append(s_input_hiddens, input_hidden, axis=0)
            
            input_service = [services] * len(input_text)
            input_services.extend(input_service)
        print(services[2:-1], s_input_hiddens.mean(), s_input_hiddens.var())
            
    # input_hiddens = np.load('/mnt/workspace/zhouyuan/L2P4DST/t5-input_hiddens.npy')
    # input_services = np.load('/mnt/workspace/zhouyuan/L2P4DST/t5-input_services.npy')
    # input_services = input_services.tolist()
    mean_name = []
    for id, service in enumerate(dataset_order):
        service_mean = np.zeros((1,512))
        length = 0
        for i in range(len(input_services)):
            if input_services[i] == service:
                service_mean += input_hiddens[i]
                length += 1
                if i+1 == len(input_services) or input_services[i+1] != service:
                    service_mean = service_mean / length
                    input_hiddens = np.append(input_hiddens, service_mean, axis=0)
                    mean_name.append(service[2:-2]+'_mean')
                    break
    input_services.extend(mean_name)
    
    
    tsne = TSNE()        
    pca=PCA(n_components=2)
    reduced_x=tsne.fit_transform(input_hiddens)
    
    plt.figure(dpi=300, figsize=(15,15))
    
    for id, service in enumerate(dataset_order):
        ser_x, ser_y = [], []
        x_mean, y_mean = 0, 0
        length = 0
        color = plt.cm.tab20(id)
        for i in range(len(reduced_x)):
            if input_services[i] == service:
                ser_x.append(reduced_x[i][0])
                ser_y.append(reduced_x[i][1])
                x_mean += reduced_x[i][0]
                y_mean += reduced_x[i][1]
                length += 1
            # elif input_services[i] == service[2:-2]+'_mean':
            #     plt.scatter(reduced_x[i][0], reduced_x[i][1], c=color, s=200, marker='*', alpha=1, edgecolors = 'r', label=service[2:-2]+'_mean')
        plt.scatter(ser_x, ser_y, c=[color]*len(ser_x), s=5, alpha=0.5, label=service[2:-2]) 
    plt.legend(loc="best")
    plt.savefig("bert-base_tsne.png")    
    plt.show()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/sgd_train.json')
    parser.add_argument('--dev', type=str, default='data/sgd_valid.json')
    parser.add_argument('--test', type=str, default='data/sgd_test.json')
    parser.add_argument('--schema', type=str, default='data/all_schema.json')
    parser.add_argument('--save_path', type=str, default='/mnt/workspace/zhouyuan/L2P4DST/best_prompt_pool_model_lr3e-2.pt')
    parser.add_argument('--task', type=str, default='MULTI')
    parser.add_argument('--select_method', type=str, default='bert')
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
    parser.add_argument('--dataset_order', type=int, default=1)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--lambda', type=float, default=0.5)


    args = parser.parse_args()
    args = args.__dict__

    main(args)