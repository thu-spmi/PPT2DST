import json
import random
import pprint
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

def t5_shift_tokens_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert (
            decoder_start_token_id is not None
    ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids

class DST_prompt_Dataset(Dataset):
    def __init__(self, args, tokenizer, data, data_id=0):
        self.args = args
        self.tokenizer = tokenizer
        self.data = data
        schemas = pd.read_json('data/all_schema.json', lines=True)
        self.schema = dict()
        for i in range(len(schemas.service_name)):
            self.schema[schemas.service_name[i]] = schemas.slots[i]

        self.soft_prompt = ''
        
        if self.args['task'] == 'MULTI': 
            for i in range(args['prompt_length']):
                self.soft_prompt += '<meta_prompt_{}>'.format(i) 
        elif self.args['task'] == 'CL':
            # self.dialogues = self.get_DST_from_dial(self.args, self.data, data['id'], self.tokenizer)
            for i in range(args['prompt_length']):
                self.soft_prompt += '<prompt_{}>'.format(data_id*100+i) 
                
        self.dialogues = []        
        for d in data:
            self.dialogues.extend(self.get_DST_from_dial(self.args, d, d['id'], self.tokenizer))

    
    def collate_fn(self, batch):
        
        history = [x[0] for x in batch]
        label_text = [x[1] for x in batch]
        slot_num = [x[2] for x in batch]

        dst_input_dict = self.tokenizer(history,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']
        
        dst_target_dict = self.tokenizer(label_text,
                                         max_length=1024,
                                         padding=True,
                                         truncation=True,
                                         return_tensors='pt')
        label = dst_target_dict['input_ids']
        dst_decoder_input_ids = t5_shift_tokens_right(label)
        # dst_decoder_input_ids[dst_decoder_input_ids[:, :] == self.tokenizer.pad_token_id] = -100
        
        return history, dst_input_ids, dst_input_mask, dst_decoder_input_ids, label, label_text, slot_num

        
    def __getitem__(self, index):
        dialogue = self.dialogues[index]
        history = "[dialogue]" + dialogue["history"] + "[state]"
        state = dialogue["state"]
        last_state = dialogue["last_state"]
        slot_num = len(self.schema[dialogue['services']])
        label = ''
        last_state_text = '[last_state]'
        for i in range(len(self.schema[dialogue['services']])):
            slots = dialogue['services'] + '-' + self.schema[dialogue['services']][i]['name']
            label += self.tokenizer.additional_special_tokens[i]
            history += self.schema[dialogue['services']][i]['name'] + ':' + self.tokenizer.additional_special_tokens[i]
            last_state_text += 'slot_{}'.format(i) + ': '
            cnt = 0
            cnt_last = 0
            for j in state.keys():
                if slots == j: 
                    label += state[j]
                    cnt = 1
                    break
            if cnt == 0: label += 'NONE'

            for j in last_state.keys():
                if slots == j: 
                    last_state_text += last_state[j] + ', '
                    cnt_last = 1
                    break
            if cnt_last == 0: last_state_text += 'NONE, '

        history = "[dialogue]" + dialogue["history"]
        history += '[prompt]' + self.soft_prompt

        return history.lower(), label.lower(), slot_num


    def __len__(self):
        return len(self.dialogues)

    def get_DST_from_dial(self, args, data, task_id, tokenizer):
        dialogues = []
        plain_history = []
        last_api = -1
        for idx_t, t in enumerate(data['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if (t['spk'] == "USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif (t['spk'] == "API-OUT"):
                pass
            elif ((t['spk'] == "SYSTEM") and idx_t != 0 and t["utt"] != ""):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif ((t['spk'] == "API") and idx_t != 0):
                slots = t["utt"].strip()
                dialogues.append({"history": " ".join(plain_history[-args['max_history']:]),
                                "reply": f'{slots} {tokenizer.eos_token}',
                                "history_reply": " ".join(
                                    plain_history[-args['max_history']:]) + f'[SOS]{slots} {tokenizer.eos_token}',
                                "spk": t["spk"],
                                "dataset": t["dataset"],
                                "dial_id": t["id"],
                                "turn_id": t["turn_id"],
                                "task_id": task_id,
                                'state': t['state'],
                                'last_state': {} if last_api == -1 else data['dialogue'][last_api]['state'],
                                "services": data['services'][0]
                                })
                last_api = idx_t
        return dialogues
    
    def get_DST_from_dial_xiaomishu(self, args, data, task_id, tokenizer):
        dialogues = []
        current_state = dict()
        for slot in self.schema[data['log'][0]['domain']]:
            current_state[slot['name']] = '无'
            
        for idx, dial in enumerate(data['log']):
            history = ''
            for dia_idx in range(idx+1):
                if dia_idx == idx:
                    history += '用户:' + data['log'][dia_idx]['caller']
                else: history += '用户:' + data['log'][dia_idx]['caller'] + ' 系统:' + data['log'][dia_idx]['called'] + ' '
            for slot in dial['bspn'].keys():
                current_state[slot] = dial['bspn'][slot]
            dialogues.append({
                "history": history,
                "state": current_state,
                "task_id": task_id,
                "services": dial['domain']
            })
        
        return dialogues
    
    
class DST_prompt_pool_Dataset(Dataset):
    def __init__(self, args, tokenizer, data, data_id=0):
        self.args = args
        self.tokenizer = tokenizer
        self.data = data
        
        if args['dataset'] == 'sgd':
            schemas = pd.read_json('data/all_schema.json', lines=True)
            self.schema = dict()
            for i in range(len(schemas.service_name)):
                self.schema[schemas.service_name[i]] = schemas.slots[i]
        elif args['dataset'] == 'xiaomishu':
            schemas = pd.read_json('data/xiaomishu_schema.json', lines=True)
            self.schema = dict()
            for i in range(len(schemas.service_name)):
                self.schema[schemas.service_name[i]] = schemas.slots[i]

        self.soft_prompt = ''
        
        if self.args['task'] == 'MULTI': 
            for i in range(args['prompt_length']):
                self.soft_prompt += '<meta_prompt_{}>'.format(i) 
        elif self.args['task'] == 'CL':
            # self.dialogues = self.get_DST_from_dial(self.args, self.data, data['id'], self.tokenizer)
            for i in range(args['prompt_length']):
                self.soft_prompt += '<prompt_{}>'.format(data_id*args['top_n']*args['prompt_length']+i) 
                
        self.dialogues = []   
        if args['dataset'] == 'sgd':     
            for d in data:
                self.dialogues.extend(self.get_DST_from_dial(self.args, d, d['task_id'], self.tokenizer))
        elif args['dataset'] == 'xiaomishu':
            for d in data:
                self.dialogues.extend(self.get_DST_from_dial_xiaomishu(self.args, d, d['task_id'], self.tokenizer))

    def sample(self, mem_size):
        samples = random.sample(self.dialogues, mem_size)
        return samples
    
    def add_sample(self, samples):
        self.dialogues.extend(samples)
    
    def collate_fn(self, batch):
        
        history = [x[0] for x in batch]
        label_text = [x[1] for x in batch]
        slot_num = [x[2] for x in batch]
        task_id = torch.tensor([x[3] for x in batch])
        input_sorted_idx = torch.stack([x[4] for x in batch], 0)

        dst_input_dict = self.tokenizer(history,
                                        max_length=1024,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        dst_input_ids = dst_input_dict['input_ids']
        dst_input_mask = dst_input_dict['attention_mask']
        
        dst_target_dict = self.tokenizer(label_text,
                                         max_length=1024,
                                         padding=True,
                                         truncation=True,
                                         return_tensors='pt')
        label = dst_target_dict['input_ids']
        dst_decoder_input_ids = t5_shift_tokens_right(label)
        # dst_decoder_input_ids[dst_decoder_input_ids[:, :] == self.tokenizer.pad_token_id] = -100
        
        return history, dst_input_ids, dst_input_mask, dst_decoder_input_ids, label, label_text, slot_num, task_id, input_sorted_idx

        
    def __getitem__(self, index):
        dialogue = self.dialogues[index]
        history = "对话: " + dialogue["history"]
        state = dialogue["state"]
        # last_state = dialogue["last_state"]
        slot_num = len(self.schema[dialogue['services']])
        # label = self.tokenizer.additional_special_tokens[0] + dialogue['services']
        label = '<extra_id_0>' + dialogue['services']
        # last_state_text = '[last_state]'
        for i in range(len(self.schema[dialogue['services']])):
            # slots = dialogue['services'] + '-' + self.schema[dialogue['services']][i]['name']
            slots = self.schema[dialogue['services']][i]['name']
            label += '<extra_id_' + str(i+1) + '>'
            # last_state_text += 'slot_{}'.format(i) + ': '
            cnt = 0
            # cnt_last = 0
            for j in state.keys():
                if slots == j: 
                    label += state[j]
                    cnt = 1
                    break
            if cnt == 0: label += '无'

            # for j in last_state.keys():
            #     if slots == j: 
            #         last_state_text += last_state[j] + ', '
            #         cnt_last = 1
            #         break
            # if cnt_last == 0: last_state_text += 'NONE, '

        # history += '[prompt]' + self.soft_prompt
        service_id = dialogue['task_id']
        input_sorted_idx = torch.arange(service_id * self.args['top_n'], (service_id+1) * self.args['top_n'])

        return history.lower(), label.lower(), slot_num, service_id, input_sorted_idx


    def __len__(self):
        return len(self.dialogues)

    def get_DST_from_dial(self, args, data, task_id, tokenizer):
        dialogues = []
        plain_history = []
        last_api = -1
        for idx_t, t in enumerate(data['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if (t['spk'] == "USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif (t['spk'] == "API-OUT"):
                pass
            elif ((t['spk'] == "SYSTEM") and idx_t != 0 and t["utt"] != ""):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif ((t['spk'] == "API") and idx_t != 0):
                slots = t["utt"].strip()
                dialogues.append({"history": " ".join(plain_history[-args['max_history']:]),
                                "reply": f'{slots} {tokenizer.eos_token}',
                                "history_reply": " ".join(
                                    plain_history[-args['max_history']:]) + f'[SOS]{slots} {tokenizer.eos_token}',
                                "spk": t["spk"],
                                "dataset": t["dataset"],
                                "dial_id": t["id"],
                                "turn_id": t["turn_id"],
                                "task_id": task_id,
                                'state': t['state'],
                                'last_state': {} if last_api == -1 else data['dialogue'][last_api]['state'],
                                "services": data['services'][0]
                                })
                last_api = idx_t
        return dialogues
    
    def get_DST_from_dial_xiaomishu(self, args, data, task_id, tokenizer):
        dialogues = []
        current_state = dict()
        for slot in self.schema[data['log'][0]['domain']]:
            current_state[slot['name']] = '无'
            
        for idx, dial in enumerate(data['log']):
            history = ''
            for dia_idx in range(idx+1):
                if dia_idx == idx:
                    history += '用户:' + data['log'][dia_idx]['caller']
                else: history += '用户:' + data['log'][dia_idx]['caller'] + ' 系统:' + data['log'][dia_idx]['called'] + ' '
            for slot in dial['bspn'].keys():
                current_state[slot] = dial['bspn'][slot]
            dialogues.append({
                "history": history,
                "state": current_state,
                "task_id": task_id,
                "services": dial['domain']
            })
        
        return dialogues