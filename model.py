import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, BertModel, AutoModel

from prompt_files.p_tuning.modeling_prompt_t5 import T5ForPromptEncDecDST
from prompt_files.t5_model.tokenization_t5 import T5Tokenizer

from heatmap_pic import cal_similarity

class T5ForPromptPool(nn.Module):
    def __init__(self, args, tokenizer, dataset_order): 
        super(T5ForPromptPool, self).__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        self.select_method = args['select_method']
        
        hidden_mean_path = 'services_mean_' + args['select_method'] + '.npy'
        services_mean_path = 'services_encoding_mean_' + args['select_method'] + '.npy'
        self.hidden_mean = np.load(hidden_mean_path)
        self.hidden_mean = torch.from_numpy(self.hidden_mean).to(torch.float32)
        self.services_mean = np.load(services_mean_path, allow_pickle=True).item()
        # gau_extrated_mean = np.load('task_extracted_info.npy', allow_pickle=True).item()
        # self.gau_extrated_mean = np.zeros((1, 384))
        # for task in dataset_order:
        #     self.gau_extrated_mean = np.append(self.gau_extrated_mean, gau_extrated_mean[task]['mean'], axis=0)
        # self.gau_extrated_mean = self.gau_extrated_mean[1:]
        
        self.id2mean = dict()
        
        # self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        # self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # self.bert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('uer/sbert-base-chinese-nli')
        self.bert_model = AutoModel.from_pretrained('uer/sbert-base-chinese-nli')
        
        for name, x in self.bert_model.named_parameters():
            x.requires_grad = False
        self.bert_model.eval()
        
        self.prompt_model = T5ForPromptEncDecDST.from_pretrained(args['model_name'])
        self.prompt_model.initialize_prompt_embedder('vocab_sample')
        self.prompt_model.set_same_prompt_pos_emb() 
        self.prompt_model.eval()
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.cos_sim = nn.CosineSimilarity()
        
        # self.pool_keys = nn.ParameterDict()
        if self.args['select_method'] == 't5':
            self.pool_keys = nn.Parameter(torch.normal(mean=0, std=0.01, size=(self.args['pool_size'], self.prompt_model.prompt_embedder.embedding_dim)), requires_grad=True)
        elif self.args['select_method'] == 'bert':
            self.pool_keys = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.args['pool_size'], 768)), requires_grad=True)
        elif self.args['select_method'] == 'bert-encoder':
            if args['dis_method'] == 'gau-bond':
                self.pool_keys = nn.Parameter(torch.from_numpy(self.gau_extrated_mean), requires_grad=False)
            else:
                self.pool_keys = nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.args['pool_size'], 384)), requires_grad=True)
                # self.pool_keys = nn.Parameter(torch.from_numpy(self.gau_extrated_mean).to(torch.float32), requires_grad=True)
        elif self.args['select_method'] == 'bert-encoder-ch':
            if args['dis_method'] == 'gau-bond':
                self.pool_keys = nn.Parameter(torch.from_numpy(self.gau_extrated_mean), requires_grad=False)
            else:
                self.pool_keys = nn.Parameter(torch.normal(mean=0, std=0.001, size=(self.args['pool_size'], 768)), requires_grad=True)
                # self.pool_keys = nn.Parameter(torch.from_numpy(self.gau_extrated_mean).to(torch.float32), requires_grad=True)
        
        self.frequency_penalty = torch.ones(self.args['pool_size'])
        self.total_num = 1
        
        self.service_prompt_num = torch.zeros(self.args['pool_size'])
        self.service_total_num = 0
            
        for name, x in self.prompt_model.named_parameters():
            if 'prompt' not in name:
                x.requires_grad = False
        # self.bert_model.eval()
        
        self.initialize_prompt_keys('random')
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def update_penalty(self):
        """Update frequency penalty"""
        self.frequency_penalty = self.frequency_penalty * self.total_num + self.service_prompt_num
        self.total_num = self.total_num + self.service_total_num
        self.frequency_penalty = self.frequency_penalty / self.total_num
        
        self.service_prompt_num = torch.zeros(self.args['pool_size'], device=self.pool_keys.device)
        self.service_total_num = 0
        
    def update_mean(self, task_id, dataset_order):
        """Update hidden_mean in order to do normalization"""
        mean = np.zeros((1,self.hidden_mean.shape[-1]))
        for i in range(task_id+1):
            mean += self.services_mean[dataset_order[i]]
        mean = mean / (task_id+1)
        self.hidden_mean = torch.from_numpy(mean).to(self.pool_keys.device).to(torch.float32)
        
    def initialize_penalty(self):
        """Initialize frequency penalty."""
        self.frequency_penalty = torch.ones(self.args['pool_size'], device=self.pool_keys.device)
        self.service_prompt_num = torch.zeros(self.args['pool_size'], device=self.pool_keys.device)
        self.total_num = 1
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def cosine_sim(self, hidden_states, dim=None, epsilon=1e-12):
        """Caculate cosin similarity between input_embedding and prompt_keys"""
        hidden_states_square_sum = torch.sum(hidden_states ** 2, dim=dim, keepdim=True) # shape: (batch_size, hidden_dim)
        prompt_key_square_sum = torch.sum(self.pool_keys ** 2, dim=dim, keepdim=True)   # shape: (pool_size, hidden_dim)
        
        norm_similarity = torch.matmul(hidden_states_square_sum, prompt_key_square_sum.t()) # shape: (batch_size, pool_size)
        similarity = torch.matmul(hidden_states, self.pool_keys.t()) * torch.rsqrt(torch.maximum(norm_similarity, torch.tensor(epsilon, device=hidden_states.device)))
        
        return similarity
                
    def initialize_prompt_keys(self, init_style):
        if init_style == 'random':
            return
        
        elif init_style == 'vocab_sample':
            if self.args['select_method'] == 't5':
                sampled_vocab_idxs = np.random.choice(self.prompt_model.vocab_size, size=self.args['max_length'], replace=True)
                sampled_vocab_mask = np.ones(sampled_vocab_idxs.shape)
                
                sampled_vocab_idxs = torch.from_numpy(sampled_vocab_idxs).unsqueeze(0)
                sampled_vocab_mask = torch.from_numpy(sampled_vocab_mask).unsqueeze(0)
                
                encoder_output = self.prompt_model(input_ids=sampled_vocab_idxs,  
                                                attention_mask=sampled_vocab_mask,
                                                use_cache=False,
                                                output_hidden_states=False,
                                                return_dict=True,
                                                only_encoder=True).encoder_last_hidden_state[:, :]
                
                encoder_mean = torch.mean(encoder_output, dim=1)
                encoder_var = torch.var(encoder_output, dim=1)
                
            elif self.args['select_method'] == 'bert':
                sampled_vocab_idxs = np.random.choice(self.bert_tokenizer.vocab_size, size=(8, 512), replace=True)
                sampled_vocab_mask = np.ones(sampled_vocab_idxs.shape)
                
                sampled_vocab_idxs = torch.from_numpy(sampled_vocab_idxs).unsqueeze(0)
                sampled_vocab_mask = torch.from_numpy(sampled_vocab_mask).unsqueeze(0)
                
                encoder_output = self.bert_model(input_ids=sampled_vocab_idxs,  
                                                attention_mask=sampled_vocab_mask,
                                                use_cache=False,
                                                output_hidden_states=False,
                                                return_dict=True,
                                                only_encoder=True).encoder_last_hidden_state[:, 0, :]
                
                encoder_mean = torch.mean(encoder_output, dim=0)
                encoder_var = torch.var(encoder_output, dim=0)
            
            for key in range(self.args['pool_size']):
                self.pool_keys.data[key] = torch.normal(encoder_mean, encoder_var)
            
    def forward(self, input_text, input, input_mask, dst_decoder_input_ids, label, device, service_name, service_id, input_task_id, input_sorted_idx, sorted_idx=None, sorted_idx_key=None):
        ## select prompt
        if self.select_method == 't5':
            input_hidden = self.prompt_model(input_ids=input,  
                                            attention_mask=input_mask,
                                            use_cache=False,
                                            output_hidden_states=False,
                                            return_dict=True,
                                            only_encoder=True).encoder_last_hidden_state[:, :]
            input_hidden = torch.mean(input_hidden, dim=1)
            
        elif self.select_method == 'bert':
            bert_input = self.bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
            bert_input.to(device)
            input_hidden = self.bert_model(**bert_input).last_hidden_state
            input_hidden = input_hidden[:,0,:]
            
        elif self.select_method == 'bert-encoder':
            encoded_input = self.bert_tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = self.bert_model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
        
        elif self.select_method == 'bert-encoder-ch':
            encoded_input = self.bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = self.bert_model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
            
        input_w_prompt, input_mask_w_prompt, s_loss, selected_keys = self.select_prompt_train(input_hidden, input_text, service_id, input_task_id, input_sorted_idx, sorted_idx, sorted_idx_key)
        input_w_prompt = input_w_prompt.to(device)
        input_mask_w_prompt = input_mask_w_prompt.to(device)
        
        ## train prompt
        outputs = self.prompt_model(input_ids=input_w_prompt, 
                                    decoder_input_ids=dst_decoder_input_ids, 
                                    attention_mask=input_mask_w_prompt,
                                    use_cache=False,
                                    output_hidden_states=False,
                                    return_dict=True,)
        lm_logits = outputs.logits
        m_loss = self.loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label.view(-1))
        loss = m_loss + self.args["lambda"]*s_loss     
        return loss, m_loss.item(), s_loss.item(), selected_keys
    
    def generate(self, input_text, input, input_mask, device, service_name, sorted_idx=None):
        if self.select_method == 't5':
            input_hidden = self.prompt_model(input_ids=input,  
                                            attention_mask=input_mask,
                                            use_cache=False,
                                            output_hidden_states=False,
                                            return_dict=True,
                                            only_encoder=True).encoder_last_hidden_state[:, :]
            input_hidden = torch.mean(input_hidden, dim=1)
            
        elif self.select_method == 'bert':
            bert_input = self.bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors="pt")
            bert_input.to(device)
            input_hidden = self.bert_model(**bert_input).last_hidden_state
            input_hidden = input_hidden[:,0,:]
            
        elif self.select_method == 'bert-encoder':
            encoded_input = self.bert_tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = self.bert_model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
        
        elif self.select_method == 'bert-encoder-ch':
            encoded_input = self.bert_tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = self.bert_model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            input_hidden = F.normalize(sentence_embeddings, p=2, dim=1)
            
        input_w_prompt, input_mask_w_prompt, s_loss, selected_keys = self.select_prompt_inference(input_hidden, input_text, service_name, sorted_idx)
        
        input_w_prompt = input_w_prompt.to(device)
        input_mask_w_prompt = input_mask_w_prompt.to(device)
        
        outputs = self.prompt_model.generate(input_w_prompt, 
                                attention_mask=input_mask_w_prompt, 
                                use_cache=False,
                                return_dict_in_generate=True,
                                max_length=100)
        return outputs, selected_keys
                  
    def select_prompt_train(self, hidden_states, input_text, service_id, input_id, input_sorted_idx, sorted_idx=None, sorted_idx_key=None, margin=1e-3):
        
        ## hidden_states shape: (batch_size, hidden_dim)
        ## pool_keys shape: (pool_size, hidden_dim)
        
        if self.args['dis_method'] == 'cosin':
            self.hidden_mean = self.hidden_mean.to(hidden_states.device)
            zero_mean_hidden = hidden_states - self.hidden_mean
            similarity = self.cosine_sim(zero_mean_hidden, dim=1)
            
            if sorted_idx == None:
                _, idx = torch.sort(similarity, dim=1, descending=True)
                idx = idx[:, :self.args['top_n']]
                sorted_idx, _ = torch.sort(idx, dim=1)
            
            temp = torch.arange(0, similarity.shape[0], device=hidden_states.device)
            temp = temp.repeat(sorted_idx.shape[1], 1).T
            similarity_value = 1 - similarity[temp.type(torch.long), sorted_idx.type(torch.long)]
            input_similarity_value = 1 + similarity[temp.type(torch.long), sorted_idx.type(torch.long)]
            # triplet_loss = torch.maximum(input_similarity_value - similarity_value + margin, torch.zeros(input_similarity_value.shape, device=hidden_states.device))
            
            input_id = input_id.repeat(self.args['top_n'], 1).T
            input_id = input_id.to(hidden_states.device)
            similarity_loss = torch.where(input_id != service_id, input_similarity_value, similarity_value)
            similarity_loss = similarity_loss.sum(dim=1).mean()
            
        elif self.args['dis_method'] == 'bce' or self.args['dis_method'] == 'bce-bond':
            self.hidden_mean = self.hidden_mean.to(hidden_states.device)
            zero_mean_hidden = hidden_states - self.hidden_mean
            temp_dot = torch.matmul(zero_mean_hidden, self.pool_keys.t()) # shape: (batch_size, pool_size)
            similarity = self.sigmoid(temp_dot)
            
            if sorted_idx == None:
                _, idx = torch.sort(similarity, dim=1, descending=True)
                idx = idx[:, :self.args['top_n']]
                sorted_idx, _ = torch.sort(idx, dim=1)
            
            temp = torch.arange(0, similarity.shape[0], device=hidden_states.device)
            temp = temp.repeat(sorted_idx.shape[1], 1).T
            similarity_value = similarity[temp.type(torch.long), sorted_idx.type(torch.long)] # shape: (batch_size, top_N)
            
            input_id = input_id.repeat(self.args['top_n'], 1).T
            input_id = input_id.to(hidden_states.device)
            temp_zeros = torch.zeros(similarity_value.shape, device=hidden_states.device)
            temp_ones = torch.ones(similarity_value.shape, device=hidden_states.device)
            temp_labels = torch.where(input_id != service_id, temp_zeros, temp_ones)
            similarity_loss = self.bce_loss(similarity_value, temp_labels)
        
        elif self.args['dis_method'] == "euclidean" or self.args['dis_method'] == "euclidean-bond":
            
            # use trainable key
            hidden_states = torch.unsqueeze(hidden_states, dim=1)
            hidden_states = hidden_states.repeat(1,self.args['pool_size'],1)
            distance = torch.sqrt(torch.sum((hidden_states - self.pool_keys) ** 2, dim=-1, keepdim=False))
            
            # # use prompt_mean as the key
            # prompt_mean = torch.zeros((self.args['pool_size'], 512), device=hidden_states.device)
            # for i in range(self.args['pool_size']):
            #     prompt_mean[i] = torch.mean(self.prompt_model.prompt_embedder.weight[i*10:(i+1)*10], dim=0)
            # prompt_hidden = hidden_states[:,:512]
            # prompt_hidden = torch.unsqueeze(prompt_hidden, dim=1)
            # prompt_hidden = prompt_hidden.repeat(1,self.args['pool_size'],1)
            # distance = torch.sqrt(torch.sum((prompt_hidden - prompt_mean) ** 2, dim=-1, keepdim=False))
             
            similarity = torch.exp(-distance)
            # similarity = distance
            
            if sorted_idx == None:
                _, idx = torch.sort(similarity, dim=1, descending=False)
                idx = idx[:, :self.args['top_n']]
                sorted_idx_key, _ = torch.sort(idx, dim=1)
                sorted_idx = torch.where(sorted_idx_key >= 11*self.args['top_n'], sorted_idx_key - 10*self.args['top_n'], sorted_idx_key % 10)
                
                
            temp = torch.arange(0, similarity.shape[0], device=hidden_states.device)
            temp = temp.repeat(sorted_idx.shape[1], 1).T
            similarity_value = similarity[temp.type(torch.long), sorted_idx_key.type(torch.long)] # shape: (batch_size, top_N)
            
            # prompt_pool
            # similarity_loss = torch.mean(torch.sum(similarity_value,dim=1))
            
            # prompt_pool_w_penalty
            input_id = input_id.repeat(self.args['top_n'], 1).T
            input_id = input_id.to(hidden_states.device)
            temp_zeros = torch.zeros(similarity_value.shape, device=hidden_states.device)
            temp_ones = torch.ones(similarity_value.shape, device=hidden_states.device)
            temp_labels = torch.where(((service_id == 0 & (input_id < (service_id+11))) | (input_id == (service_id+10))), temp_ones, temp_zeros)
            temp_values = torch.where(input_id != service_id, temp_zeros, similarity_value)
            
            # update as the 
            # use trainable key
            similarity_loss = self.bce_loss(similarity_value, temp_labels)
            # similarity_loss = torch.mean(torch.sum(1 - similarity_value, dim=1))
            
            # # use prompt_mean as key
            # similarity_loss = torch.tensor(0)
        
        elif self.args['dis_method'] == 'gau-bond':
            hidden_states = torch.unsqueeze(hidden_states, dim=1)
            hidden_states = hidden_states.repeat(1,self.args['pool_size'],1)
            distance = torch.sqrt(torch.sum((hidden_states - self.pool_keys) ** 2, dim=-1, keepdim=False))
            similarity = torch.exp(-distance)
            
            if sorted_idx == None:
                _, idx = torch.sort(similarity, dim=1, descending=True)
                idx = idx[:, :self.args['top_n']]
                sorted_idx, _ = torch.sort(idx, dim=1)
            
            similarity_loss = torch.tensor(0)
        
        for b_id, h in enumerate(hidden_states):
            input_text[b_id] += '[prompts]'
            for key in sorted_idx[b_id]:
                for i in range(self.args['prompt_length']):
                    input_text[b_id] += '<prompt_{}>'.format(key*self.args['prompt_length']+i) 
        
        dst_input_dict = self.tokenizer(input_text,
                                max_length=1024,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']
        
        b_selected_keys = sorted_idx.cpu().detach().numpy().tolist()
        self.service_total_num += len(input_text)
        for key in sorted_idx:
            self.service_prompt_num[key] += 1
        
        return dst_input_ids, dst_input_mask, similarity_loss, b_selected_keys
    
    def select_prompt_inference(self, hidden_states, input_text, service_name, sorted_idx=None):
        s_loss = 0
        b_selected_keys = []
        
        if self.args['dis_method'] == 'cosin':
            self.hidden_mean = self.hidden_mean.to(hidden_states.device)
            similarity = self.cosine_sim(hidden_states - self.hidden_mean, dim=1)
        elif self.args['dis_method'] == 'bce' or self.args['dis_method'] == 'bce-bond':
            # temp_dot = torch.matmul(hidden_states - self.hidden_mean, self.pool_keys.t())
            temp_dot = torch.matmul(hidden_states - self.hidden_mean, self.pool_keys.t())
            similarity = self.sigmoid(temp_dot)
        elif self.args['dis_method'] == 'euclidean' or self.args['dis_method'] == 'euclidean-bond' or self.args['dis_method'] == 'gau-bond':
            prompt_mean = torch.zeros((self.args['pool_size'], 512), device=hidden_states.device)
            for i in range(self.args['pool_size']):
                prompt_mean[i] = torch.mean(self.prompt_model.prompt_embedder.weight[i*10:(i+1)*10], dim=0)
            prompt_hidden = hidden_states[:,:512]
            
            # # use prompt_mean as the key
            # prompt_hidden = torch.unsqueeze(prompt_hidden, dim=1)
            # prompt_hidden = prompt_hidden.repeat(1,self.args['pool_size'],1)
            # distance = torch.sqrt(torch.sum((prompt_hidden - prompt_mean) ** 2, dim=-1, keepdim=False))
            
            # use trainable key
            hidden_states = torch.unsqueeze(hidden_states, dim=1)
            hidden_states = hidden_states.repeat(1,self.args['pool_size'],1)
            distance = torch.sqrt(torch.sum((hidden_states - self.pool_keys) ** 2, dim=-1, keepdim=False))
            similarity = torch.exp(-distance) 
        
        if sorted_idx == None:
            if 'bond' in self.args['dis_method']:
                value, idx = torch.topk(similarity, k=1, dim=1, sorted=False)
                idx = torch.argmax(similarity, dim=1) // self.args['top_n']
                idx = idx.unsqueeze(dim=1)
                sorted_idx = idx.repeat(1, self.args['top_n'])
                incre_temp = torch.arange(0, self.args['top_n'], device=sorted_idx.device)
                sorted_idx = sorted_idx*self.args['top_n'] + incre_temp    
                
                value = value.repeat(1, self.args['top_n'])
            else:
                value, idx = torch.topk(similarity, k=self.args['top_n'], dim=1, sorted=False)
                sorted_idx, _ = torch.sort(idx, dim=1)
                sorted_idx = torch.where(sorted_idx >= 11*self.args['top_n'], sorted_idx - 10*self.args['top_n'], sorted_idx % 10)
                sorted_idx, _ = torch.sort(sorted_idx, dim=1)
        else:
            temp = torch.arange(0, similarity.shape[0], device=hidden_states.device)
            temp = temp.repeat(sorted_idx.shape[1], 1).T
            value = similarity[temp.type(torch.long), sorted_idx.type(torch.long)]
        
        for b_id, h in enumerate(hidden_states):
            input_text[b_id] += '[prompts]'
            for key in sorted_idx[b_id]:
                for i in range(self.args['prompt_length']):
                    input_text[b_id] += '<prompt_{}>'.format(key*self.args['prompt_length']+i) 
        
        dst_input_dict = self.tokenizer(input_text,
                                max_length=1024,
                                padding=True,
                                truncation=True,
                                return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']
        
        b_selected_keys = sorted_idx.cpu().detach().numpy().tolist()
        
        return dst_input_ids, dst_input_mask, (1-value).mean(), b_selected_keys