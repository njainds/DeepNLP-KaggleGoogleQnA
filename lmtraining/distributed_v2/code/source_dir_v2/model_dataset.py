import pandas as pd
import numpy as np
import torch
from transformers import *
from sklearn.utils import shuffle
import random
import html
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GroupKFold
from math import floor, ceil
from ml_stratifiers import MultilabelStratifiedKFold
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, tokenizer=None, target_cols = None, max_len=128, content="Question_Answer",augment = True,labeled=True):
        self.df = df
        self.max_len=max_len
        self.content = content
        self.tokenizer = tokenizer
        self.target_columns = target_cols

    def __len__(self):
        return len(self.df)

    def _get_text(self, index):
        row = self.df.iloc[index]
        title = row.question_title if row.question_title is not None else None
        body = row.question_body if row.question_body is not None else None
        answer = row.answer if row.answer is not None else None
        return title, body, answer

    def select_tokens(self, tokens, max_num):
        if len(tokens) < max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            idx = random.randint(0,len(tokens)-num_remove-1)
            return tokens[:idx] + tokens[idx+num_remove:]
        else:
            return tokens[:max_num//2] + tokens[-(max_num-max_num//2):]


    def trim_input(self, title, body, answer, max_seq_len = 128, t_max_len =30, q_max_len = int((128-30-4)/2), a_max_len = 128-30-4-int((128-30-4)/2), num_token = 4):
        body = html.unescape(body)
        answer = html.unescape(answer)
        title = html.unescape(title)

        t = self.tokenizer.tokenize(title)
        q = self.tokenizer.tokenize(body)
        a = self.tokenizer.tokenize(answer)

        t_len = len(t)
        q_len = len(q)
        a_len = len(a)

        if (t_len + q_len + a_len + num_token)>max_seq_len:
            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
                q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
            else:
                t_new_len = t_max_len

            if a_max_len > a_len:
                a_new_len = a_len
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len

            q_new_len = int(q_new_len)
            a_new_len = int(a_new_len)
            t_new_len = int(t_new_len)

            if int(t_new_len+ a_new_len+q_new_len +num_token) >max_seq_len:
                print("new seq len is %d ut it must be less than or equal to %d" %((t_new_len+ q_new_len +a_new_len +num_token),max_seq_len))
            if t_len > t_new_len:
                t = t[:int(t_new_len//4)] + t[int(t_len - t_new_len+ t_new_len//4): ]
            else:
                t = t[:int(t_new_len)]

            if q_len > q_new_len:
                q = q[:int(q_new_len//4)] + q[int(q_len - int(q_new_len) + q_new_len//4): ]
            else:
                q = q[:int(q_new_len)]

            if a_len > a_new_len:
                a = a[:int(a_new_len//4)] + a[int(a_len - int(a_new_len) + a_new_len//4): ]
            else:
                a = a[:int(a_new_len)]

        if (len(t) + len(q)+len(a) + num_token > max_seq_len):
            more_token = len(t) + len(q) +len(a) + num_token - max_seq_len
            a = a[:(len(a) - more_token)]

        return t, q, a

    def get_token_ids(self, title, body, answer):
        num_token = 4
        t_max_len = 30
        a_max_len = floor(self.max_len - t_max_len - num_token)/2
        q_max_len = ceil(self.max_len - t_max_len - num_token)/2
        t_tokens, q_tokens, a_tokens = self.trim_input(title,body, answer,max_seq_len=self.max_len, t_max_len=t_max_len, q_max_len=q_max_len,a_max_len=a_max_len, num_token=num_token)
        tokens = ["[CLS]"] + t_tokens + ["[SEP]"] + q_tokens + ["[SEP]"] + a_tokens + ["[SEP]"]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0]*(self.max_len - len(token_ids))
        ids = torch.tensor(token_ids)
        seg_ids = self.get_seg_ids(ids)
        attention_mask = (ids>0).type(torch.long)
        return ids, seg_ids, attention_mask

    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        first_sep = True
        seg_idx = 0
        for i,e in enumerate(ids):
            seg_ids[i] = seg_idx
            if e == self.tokenizer.sep_token_id:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids==0)
        seg_ids[pad_idx]  = 0
        return seg_ids

    def get_label(self, index):
        row = self.df.iloc[index]
        return torch.tensor(row[self.target_columns].values.astype(np.float32))


    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        seg_ids = torch.stack([x[1] for x in batch])

        if self.labeled:
            labels = torch.stack([x[2] for x in batch])
            return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids
