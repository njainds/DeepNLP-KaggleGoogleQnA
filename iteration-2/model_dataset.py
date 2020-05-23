import pandas as pd
import numpy as np
import torch
from transformers import *
from sklearn.utils import shuffle
import random
import html
import nlpaug.flow as naf
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GroupKFold
from math import floor, ceil
from ml_stratifiers import MultilabelStratifiedKFold
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

## Define Arguments for the script

parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument('-data_path',      type=str, default= 'C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/train.csv', required=False, help='specify the path of train.csv')
parser.add_argument('-test_data_path', type=str, default= 'C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/test.csv', required=False, help='specify the path of test.csv')
parser.add_argument('-content',        type=str, default='Question', required=False, help='specify the content of token')
parser.add_argument('--n_splits',      type=int, default=5, required=False, help='specify the # of folds')
parser.add_argument('--seed',          type=int, default=42, required=False, help='specify the seed')
parser.add_argument('-save_path',      type=str, default= 'C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/processed/', required=False, help='specify the path for saving processed data')
parser.add_argument('--test_fold',     type=int, default=0, required=False, help='specify the # of folds for test loader')
parser.add_argument('--batch_size',    type=int, default=8, required=False, help='specify the batch size for train loader')
parser.add_argument('--val_batch_size',type=int, default=8, required=False, help='specify the batch size for test loader')
parser.add_argument('--model_type',    type=str, default='bert-base-uncased', required=False, help='specify the # type of model')
parser.add_argument('--num_workers',   type=int, default=0, required=False, help='specify the num_workers for testing dataloader')


sep_token_id = 102

df = pd.read_csv(os.getcwd() + '/data/sample_submission.csv')
question_target_columns = [col for col in df.columns.tolist()[1:] if 'question_' in col]
answer_target_columns = [col for col in df.columns.tolist()[1:] if 'answer_' in col]
target_columns = question_target_columns + answer_target_columns

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, df,  host_encoder=None, category_encoder=None, max_len=512, model_type = 'bert-base-cased', content="Question",augment = True,labeled=True, train_mode=True, extra_token=False):
        self.df = df
        self.max_len=max_len
        self.host_encoder=host_encoder
        self.category_encoder = category_encoder
        self.train_mode = train_mode
        self.model_type = model_type
        self.labeled  = labeled
        self.extra_token = extra_token
        self.augment = augment
        self.content = content

        if ((self.model_type == 'bert-base-uncased') or (self.model_type == 'bert-large-uncased')):
            self.tokenizer = BertTokenizer.from_pretrained(self.model_type, additional_special_tokens = ["[UNK]","[SEP]","[PAD]","[CLS]","[MASK]"])
        elif ((self.model_type == 'bert-base-cased') or (self.model_type == 'bert-large-cased')):
            self.tokenizer = BertTokenizer.from_pretrained(self.model_type, additional_special_tokens = ["[UNK]","[SEP]","[PAD]","[CLS]","[MASK]"])
        elif ((self.model_type == 'xlnet-base-cased') or (self.model_type == 'xlnet-large-cased')):
            self.tokenizer = XLNetTokenizer.from_pretrained(self.model_type, additional_special_tokens = ["[UNK]","[SEP]","[PAD]","[CLS]","[MASK]"])
        elif (self.model_type == 'roberta-base'):
            add_tokens = ["[TITLE]","[BODY]","[ANSWER]","[CATEGORY]","[DOMAIN]","[HOST]",\
                          "[category:LIFE_ARTS]","[category:CULTURE]","[category:SCIENCE]","[category:STACKOVERFLOW]","[category:TECHNOLOGY]",\
                          "[domain:stackexchange]","[domain:stackoverflow]","[domain:askubuntu]","[domain:serverfault]","[domain:superuser]","[domain:mathoverflow]"\
                          ]+ list(df.host.unique().tolist())
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_type)
            self.tokenizer.cls_token = "[CLS]"
            self.tokenizer.sep_token = "[SEP]"
            num_added_tokens = self.tokenizer.add_tokens(add_tokens)
            print("Number of Tokens added: ", num_added_tokens)
        elif (self.model_type == "gpt2"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, additional_special_tokens = ["[UNK]","[SEP]","[PAD]","[CLS]","[MASK]"])
        else:
            raise NotImplementedError

        self.translation_title_rate = 0.5
        self.translation_body_rate = 0.5
        self.translation_answer_rate = 0.5
        self.translation_single_language = 0.25
        self.random_select_date = 0.1

        if host_encoder is not None:
            transformed = host_encoder.transform(self.df['host'])
            self.df['host'] = transformed.tolist()
            self.df['host']  = self.df['host'].apply(lambda x: np.array(x))

        if category_encoder is not None:
            transformed = category_encoder.transform(self.df['category'])
            self.df['category'] = transformed.tolist()
            self.df['category']  = self.df['category'].apply(lambda x: np.array(x))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids, seg_ids  = self.get_token_ids(row, index)
        if self.labeled:
            labels = self.get_label(row)
            if self.extra_token:
                label_category = torch.tensor(row.category)
                label_host = torch.tensor(row.host)
                return token_ids, seg_ids, labels, label_category, label_host
            return token_ids, seg_ids, labels
        return token_ids, seg_ids

    def __len__(self):
        return len(self.df)

    def augmentation(self, text, insert=False, substitute=False, swap=True, delete=True):

        augs = []

        if insert:
            # aug = naw.ContextualWordEmbsAug(
            #     model_path=self.model_type, action="insert", device='cuda')
            # wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
            aug = naw.WordEmbsAug(
                model_type='word2vec',
                model_path='/C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/helpers/word2vec/GoogleNews-vectors-negative300.bin',
                action="insert")
            augs.append(aug)

        if substitute:
            # aug = naw.ContextualWordEmbsAug(
            #     model_path=self.model_type, action="substitute", device='cuda')
            # aug = naw.WordEmbsAug(
            #     model_type='word2vec', model_path='/media/jionie/my_disk/Kaggle/Google_Quest_Answer/model/word2vec/GoogleNews-vectors-negative300.bin',
            #     action="substitute")
            aug_sub = naw.SynonymAug(aug_src='wordnet')
            augs.append(aug_sub)
            # text = aug.augment(text)

        if swap:
            aug_swap = naw.RandomWordAug(action="swap")
            augs.append(aug_swap)
            # text = aug.augment(text)

        if delete:
            aug_del = naw.RandomWordAug()
            augs.append(aug_del)
            # text = aug.augment(text)

        aug = naf.Sometimes(augs, aug_p=0.5, pipeline_p=0.5)
        # print("before aug:", text)
        text = aug.augment(text, n=1)
        # print("after aug:", text)

        return text

    def select_tokens(self, tokens, max_num):
        if len(tokens) < max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            idx = random.randint(0,len(tokens)-num_remove-1)
            return tokens[:idx] + tokens[idx+num_remove:]
        else:
            return tokens[:max_num//2] + tokens[-(max_num-max_num//2):]

    def trim_input_single_content(self, title, content, max_seq_len = 512, t_max_len =30, c_max_len = 512-30-4, num_token = 3):
        if self.augment:
            title = self.augmentation(title, insert=False, substitute=False, swap = False, delete= True)
            content = self.augmentation(content, insert=False, substitute=False, swap = False, delete= True)
        t = self.tokenizer.tokenize(title)
        c = self.tokenizer.tokenize(content)

        t_len = len(t)
        c_len = len(c)

        if (t_len + c_len + num_token)>max_seq_len:
            if t_len<t_max_len:
                t_new_len = t_len
                c_max_len = c_max_len + floor((t_max_len-t_len)/2)
            else:
                t_new_len = t_max_len

            if c_len < c_max_len:
                c_new_len = c_len
            else:
                c_new_len = c_max_len

            if t_new_len+ c_new_len +num_token >max_seq_len:
                print("new seq len is %d ut it must be less than or equal to %d" %((t_new_len+ c_new_len +num_token),max_seq_len))

            if self.augment:
                if random.random() < self.random_select_date:
                    if t_len>t_new_len:
                        t_start = np.random.randint(0,t_len-t_new_len)
                        t = t[t_start:(t_start+t_new_len)]
                    else:
                        t_start = 0
                        t = t[t_start:(t_start + t_new_len)]

                    if c_len>c_new_len:
                        c_start = np.random.randint(0,c_len-c_new_len)
                        c = c[c_start:(c_start+c_new_len)]
                    else:
                        c_start = 0
                        c = c[c_start:(c_start + c_new_len)]
                else:
                    if t_len > t_new_len:
                        t = t[:t_new_len//4] + t[t_len - t_new_len+ t_new_len//4: ]
                    else:
                        t = t[:t_new_len]

                    if c_len > c_new_len:
                        c = c[:c_new_len//4] + c[c_len - c_new_len+ c_new_len//4: ]
                    else:
                        c = c[:c_new_len]

            else:
                if t_len > t_new_len:
                    t = t[:t_new_len//4] + t[t_len - t_new_len+ t_new_len//4: ]
                else:
                    t = t[:t_new_len]

                if c_len > c_new_len:
                    c = c[:c_new_len//4] + c[c_len - c_new_len+ c_new_len//4: ]
                else:
                    c = c[:c_new_len]
                # some bad cases
        if (len(t) + len(c) + num_token > max_seq_len):
            more_token = len(t) + len(c) + num_token - max_seq_len
            c = c[:(len(c) - more_token)]

        return t, c

    def trim_input(self, title, question, answer, max_seq_len = 512, t_max_len =30, q_max_len = int((512-30-4)/2), a_max_len = 512-30-4-int((512-30-4)/2), num_token = 4):
        question = html.unescape(question)
        answer = html.unescape(answer)
        title = html.unescape(title)

        if self.augment:
            title = self.augmentation(title, insert=False, substitute=False, swap = False, delete= True)
            question = self.augmentation(question, insert=False, substitute=False, swap = False, delete= True)
            answer = self.augmentation(answer, insert=False, substitute=False, swap = False, delete= True)
        t = self.tokenizer.tokenize(title)
        q = self.tokenizer.tokenize(question)
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

            if self.augment:

                if random.random() < self.random_select_date:
                    if t_len>t_new_len:
                        t_start = np.random.randint(0,t_len-t_new_len)
                        t = t[t_start:(t_start+t_new_len)]
                    else:
                        t_start = 0
                        t = t[t_start:(t_start + t_new_len)]

                    if q_len>q_new_len:
                        q_start = np.random.randint(0,q_len-q_new_len)
                        q = q[q_start:(q_start+q_new_len)]
                    else:
                        q_start = 0
                        q = q[q_start:(q_start + q_new_len)]

                    if a_len>a_new_len:
                        a_start = np.random.randint(0,a_len-a_new_len)
                        a = a[a_start:(a_start+a_new_len)]
                    else:
                        a_start = 0
                        a = a[a_start:(a_start + a_new_len)]

                else:
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

            else:
                if t_len > t_new_len:
                    t = t[:int(t_new_len//4)] + t[int(t_len - t_new_len+ t_new_len//4): ]
                else:
                    t = t[:int(t_new_len)]

                if q_len > q_new_len:
                    q = q[:int(q_new_len//4)] + q[int(q_len - int(q_new_len)+ q_new_len//4): ]
                else:
                    q = q[:int(q_new_len)]

                if a_len > a_new_len:
                    a = a[:int(a_new_len//4)] + a[int(a_len - int(a_new_len) + a_new_len//4): ]
                else:
                    a = a[:int(a_new_len)]

                # some bad cases
        if (len(t) + len(q)+len(a) + num_token > max_seq_len):
            more_token = len(t) + len(q) +len(a) + num_token - max_seq_len
            a = a[:(len(a) - more_token)]

        return t, q, a

    def get_token_ids(self, row, index):
        if self.extra_token:
            num_token = 6
        else:
            num_token = 4
        if self.content == "Question":
            num_token -= 1
            t_max_len = 30
            q_max_len = self.max_len - t_max_len -num_token
            a_max_len  =0
            t_tokens, c_tokens = self.trim_input_single_content(row.question_title, row.question_body, max_seq_len=self.max_len,t_max_len = t_max_len, c_max_len=q_max_len, num_token=num_token)
        elif self.content == "Answer":
            num_token -= 1
            t_max_len = 30
            a_max_len = self.max_len - t_max_len - num_token
            q_max_len = 0
            t_tokens, c_tokens = self.trim_input_single_content(row.question_title, row.answer,max_seq_len=self.max_len, t_max_len=t_max_len,c_max_len=a_max_len, num_token=num_token)
        elif self.content == "Question_Answer":
            #num_token = -1
            t_max_len = 30
            a_max_len = floor(self.max_len - t_max_len - num_token)/2
            q_max_len = ceil(self.max_len - t_max_len - num_token)/2
            t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title,row.question_body, row.answer,max_seq_len=self.max_len, t_max_len=t_max_len, q_max_len=q_max_len,a_max_len=a_max_len, num_token=num_token)
        else:
            raise NotImplementedError

        if ('bert' in self.model_type) or ('xlnet' in self.model_type) or ('roberta' in self.model_type):
            if self.content == "Question_Answer":
                if self.extra_token:
                    tokens = ["[CLS]"] + ["[CLS]"] + ["[CLS]"] + t_tokens + ["[SEP]"] + q_tokens + ["[SEP]"] + a_tokens + ["[SEP]"]
                else:
                    tokens = ["[CLS]"] + t_tokens + ["[SEP]"] + q_tokens + ["[SEP]"] + a_tokens + ["[SEP]"]
            elif self.content == "Question" or self.content == "Answer":
                if self.extra_token:
                    tokens = ["[CLS]"] + ["[CLS]"] + ["[CLS]"] + t_tokens + ["[SEP]"] + c_tokens + ["[SEP]"]
                else:
                    tokens = ["[CLS]"] + t_tokens + ["[SEP]"] + c_tokens + ["[SEP]"]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0]*(self.max_len - len(token_ids))
        ids = torch.tensor(token_ids)
        seg_ids = self.get_seg_ids(ids)
        return ids, seg_ids

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

    def get_label(self, row):
        if self.content == "Question":
            return torch.tensor(row[question_target_columns].values.astype(np.float32))
        elif self.content == "Answer":
            return torch.tensor(row[answer_target_columns].values.astype(np.float32))
        elif self.content == "Question_Answer":
            return torch.tensor(row[target_columns].values.astype(np.float32))
        else:
            raise NotImplementedError

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        seg_ids = torch.stack([x[1] for x in batch])

        if self.labeled:
            labels = torch.stack([x[2] for x in batch])
            if self.extra_token:
                category_labels = torch.stack([x[3] for x in batch])
                host_labels = torch.stack([x[4] for x in batch])
                return token_ids, seg_ids, labels, category_labels, host_labels
            else:
                return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids

def get_test_loader(data_path = "C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/test.csv", max_len=512, model_type='bert-model-uncased', content='Question', batch_size=8, extra_token=True):
    test_df = pd.read_csv(data_path)
    ds_test = QuestDataset(test_df, None, None, max_len, model_type, content=content, train_mode=False, labeled=False, augment=False, extra_token=extra_token)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle = False, num_workers = 0, collate_fn = ds_test.collate_fn,drop_last = False)
    loader.num = len(test_df)
    return loader

def get_train_val_split(data_path = "C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/train.csv",
                        save_path = "C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/processed/",
                        n_splits =5, seed = 42, split = "GroupKFold"):
    os.makedirs(save_path + 'split/', exist_ok=True)
    df = pd.read_csv(data_path, encoding = 'utf8')

    if split == 'GroupKFold':
        df = shuffle(df, random_state = seed)
        kf = GroupKFold(n_splits=n_splits).split(X = df.question_body, groups=df.question_body)
    elif split =='MultilabelStratifiedKFold':
        kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle =True).split(df.question_body, df[target_columns].values)
    else:
        raise NotImplementedError

    for fold, (train_idx, val_idx) in enumerate(kf):
        df_train = df.iloc[train_idx]
        df_val  = df.iloc[val_idx]

        df_train.to_csv(save_path + 'split/train_fold_%s_seed_%s.csv' % (fold, seed), index=False)
        df_val.to_csv(save_path + 'split/val_fold_%s_seed_%s.csv' % (fold, seed), index=False)
    return

def get_train_val_loaders(train_data_path = "C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/processed/split/train_fold_0_seed_42.csv",
                            val_data_path="C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/processed/split/val_fold_0_seed_42.csv",
                            category_encoder = None, host_encoder = None, batch_size = 4, val_batch_size=4, max_len = 512, content = "Question", model_type = 'bert_base-uncased',
                            num_workers = 2, augment=True, extra_token=True):
    train_df = pd.read_csv(train_data_path, encoding='utf8')
    val_df = pd.read_csv(val_data_path, encoding='utf8')
    print("######################################content is :", content)
    ds_train = QuestDataset(train_df,host_encoder, category_encoder, max_len, model_type, content=content,  train_mode=True, labeled=True, augment=augment, extra_token=extra_token)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(ds_train)

    ds_val = QuestDataset(val_df, host_encoder, category_encoder, max_len, model_type, content=content,
                                train_mode=True, labeled=True, augment=augment, extra_token=extra_token)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True,num_workers=num_workers, collate_fn=ds_train.collate_fn,drop_last=True)
    val_loader.num = len(ds_val)
    val_loader.df = ds_val

    return train_loader, val_loader, ds_train.tokenizer

##Test cases

def test_train_val_split(data_path, save_path, n_splits, seed):
    print("Testing train-test splitting")
    print("data path: ", data_path)
    print("data path: ", save_path)
    print("n_splits: ", n_splits)
    print("seed: ", seed)
    get_train_val_split(data_path, save_path, n_splits, seed)
    if os.path.exists(save_path + 'split/val_fold_0_seed_42.csv'):
        print("test successfull")
    else:
        print("Error")


def test_train_loader(train_data_path = "C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/processed/split/train_fold_0_seed_42.csv",
                            val_data_path="C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/processed/split/val_fold_0_seed_42.csv",
                            category_encoder = None, host_encoder = None, batch_size = 4, val_batch_size = 4, max_len = 512, content = "Question", model_type = 'bert_base-uncased',
                            num_workers = 2, augment=True, extra_token=True):
    train_loader, val_loader, tokenizer = get_train_val_loaders(train_data_path=train_data_path, val_data_path=val_data_path, host_encoder=host_encoder, category_encoder=category_encoder, model_type=model_type, content=content, batch_size=batch_size, val_batch_size=val_batch_size,num_workers=num_workers, extra_token=extra_token)

    if extra_token:
        for ids, seg_ids, labels, category_labels, host_labels in train_loader:
            print("------------------------testing train loader with extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("labels: ", labels)
            print("category_labels shape: ", category_labels.shape)
            print("host_labels shape: ", host_labels.shape)
            print("category_labels: ", category_labels)
            print("host_labels: ", host_labels)
            print("------------------------testing train loader finished----------------------")
            break

        for ids, seg_ids, labels, category_labels, host_labels in val_loader:
            print("------------------------testing val loader with extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("category_labels shape: ", category_labels.shape)
            print("host_labels shape: ", host_labels.shape)
            print("labels: ", labels)
            print("category_labels: ", category_labels)
            print("host_labels: ", host_labels)
            print("------------------------testing val loader finished----------------------")
            break

    else:
        for ids, seg_ids, labels in train_loader:
            print("------------------------testing train loader without extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("labels: ", labels)
            print("------------------------testing train loader finished----------------------")
            break

        for ids, seg_ids, labels in val_loader:
            print("------------------------testing val loader without extra_token----------------------")
            print("ids:", ids)
            print("seg_ids (numpy): ", seg_ids.numpy())
            print("labels: ", labels)
            print("------------------------testing val loader finished----------------------")
            break


def test_test_loader(data_path="C:/Users/admin/Documents/Nitin/mycodes/kaggle_google_quest_qna/data/test.csv", model_type="bert-base-uncased", content="Question", batch_size=4, extra_token=True):
    loader = get_test_loader(data_path=data_path,model_type=model_type, content=content, batch_size=batch_size, extra_token=extra_token)
    for ids, seg_ids in loader:
        print("------------------------testing test loader----------------------")
        print("ids: ", ids)
        print("seg_ids (numpy): ", seg_ids.numpy())
        print("------------------------testing test loader finished----------------------")
        break

if __name__ == "__main__":
    args = parser.parse_args()
    print("check1")
    train_df = pd.read_csv(args.data_path, encoding='utf8')
    test_df = pd.read_csv(args.test_data_path, encoding='utf8')
    host_encoder = LabelBinarizer()
    host_encoder.fit(list(set(train_df['host'].unique().tolist() + test_df['host'].unique().tolist())))
    category_encoder = LabelBinarizer()
    category_encoder.fit(list(set(train_df['category'].unique().tolist() + test_df['category'].unique().tolist())))

    # test getting train val splitting
    test_train_val_split(args.data_path, args.save_path,args.n_splits,args.seed)

    train_data_path = args.save_path + '/split/train_fold_%s_seed_%s.csv' % (args.test_fold, args.seed)
    val_data_path = args.save_path + '/split/val_fold_%s_seed_%s.csv' % (args.test_fold, args.seed)

    print("################### Data read ##################################")

    test_train_loader(train_data_path=train_data_path, val_data_path=val_data_path, category_encoder=category_encoder,
                      host_encoder=host_encoder, batch_size=args.batch_size, content=args.content,
                      model_type=args.model_type, num_workers=args.num_workers, extra_token=True)

    test_train_loader(train_data_path=train_data_path,val_data_path=val_data_path,category_encoder=category_encoder, host_encoder=host_encoder, batch_size=args.batch_size, content=args.content,model_type=args.model_type,num_workers=args.num_workers, extra_token=False)
    print("check")
    test_test_loader(data_path=args.test_data_path, model_type=args.model_type, content=args.content, batch_size=args.batch_size, extra_token=True)

    test_test_loader(data_path=args.test_data_path, model_type=args.model_type, content=args.content, batch_size=args.batch_size, extra_token=False)




