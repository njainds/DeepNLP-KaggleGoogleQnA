from transformers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuestNet(nn.Module):
    def __init__(self,  model_type = 'bert-base-uncased', tokenizer=None, n_classes=30, n_category_classes = 5, \
                 n_host_classes = 64, hidden_layers = [-1,-3,-5,-7], extra_token = False):
        super(QuestNet, self).__init__()
        self.model_type = model_type
        self.model_name = "Questmodel"
        self.hidden_layers  = hidden_layers
        self.extra_token = extra_token

        if model_type == 'bert-base-uncased':
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob = 0.1, output_hidden_states = True)
            self.hidden_size = 768
        elif model_type == 'bert-large-uncased':
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0.1, output_hidden_states=True)
            self.hidden_size = 1024
        elif model_type == 'bert-base-cased':
            self.bert_model = BertModel.from_pretrained(model_type, hidden_dropout_prob=0.1, output_hidden_states=True)
            self.hidden_size = 768
        elif model_type == 'xlnet-base-cased':
            self.xlnet_model = XLNetModel.from_pretrained(model_type, dropout=0.1, output_hidden_states=True)
            self.hidden_size = 1024
        elif model_type == 'roberta-base':
            self.roberta_model = RobertaModel.from_pretrained(model_type, hidden_dropout_prob=0.1, output_hidden_states=True)
            self.roberta_model.resize_token_embeddings(len(tokenizer))
            self.hidden_size = 768
        elif model_type == 'gpt-2':
            self.gpt2_model = GPT2Model.from_pretrained(model_type, initializer_range=0, output_hidden_states=True)
            self.hidden_size = 768
        else:
            raise NotImplementedError

        self.fc_1 = nn.Linear((self.hidden_size)*len(hidden_layers), self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, n_classes)

        if extra_token:
            self.fc_1_category = nn.Linear((self.hidden_size)*len(hidden_layers), self.hidden_size)
            self.fc_category = nn.Linear(self.hidden_size, n_category_classes)
            self.fc_1_host = nn.Linear((self.hidden_size) * len(hidden_layers), self.hidden_size)
            self.fc_host = nn.Linear(self.hidden_size, n_host_classes)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        #self.prelu = nn.PRELU()
        self.tanh = nn.Tanh()

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def get_hidden_state_by_index(self, hidden_states, index):
        for i in range(len(self.hidden_layers)):
            if i == 0:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer][:, index]
                fuse_hidden = torch.unsqueeze(hidden_state,dim = -1)
            else:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer][:, index]
                h = torch.unsqueeze(hidden_state,-1)
                fuse_hidden = torch.cat([fuse_hidden,h],dim=-1)
        fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0],-1)
        return fuse_hidden

    def get_hidden_state_by_mean(self,hidden_states):
        for i in range(self.hidden_layers):
            if i == 0:
                hidden_layer = self.hidden_layers[i]
                hidden_state = torch.mean(hidden_states[hidden_layer], dim =1)
                fuse_hidden = torch.unsqeeze(hidden_state, dim=-1)
            else:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer][:, index]
                h = torch.unsqueeze(hidden_state, -1)
                fuse_hidden = torch.cat([fuse_state, h], dim=-1)
        fuse_hidden = fuse_hidden.reshape(fuse_hidden.shape[0], -1)
        return fuse_hidden

    def get_logits_by_random_dropout(self, fuse_hidden, fc_1, fc):
        h = self.relu(fc_1(fuse_hidden))
        for j, dropout in enumerate(self.dropouts):
            if j == 0:
                logit = fc(dropout(h))
            else:
                logit += fc(dropout(h))
        return logit/len(self.dropouts)

    def forward(self, ids, seg_ids):
        attention_mask = (ids>0)

        if 'roberta' in self.model_type:
            outputs = self.roberta_model(input_ids = ids, attention_mask = attention_mask)
            hidden_states = outputs[2]

            fuse_hidden = self.get_hidden_state_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)

            if self.extra_token:
                fuse_hidden_catg = self.get_hidden_state_by_index(hidden_states, 1)
                logits_catg = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_category, self.fc_category)

                fuse_hidden_host = self.get_hidden_state_by_index(hidden_states, 2)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_host, self.fc_host)

        
        elif 'bert-' in self.model_type:
            outputs = self.bert_model(input_ids = ids, attention_mask = attention_mask, token_type_ids = seg_ids)
            hidden_states = outputs[2]

            fuse_hidden = self.get_hidden_state_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)

            if self.extra_token:
                fuse_hidden_catg = self.get_hidden_state_by_index(hidden_states, 1)
                logits_catg = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_category, self.fc_category)

                fuse_hidden_host = self.get_hidden_state_by_index(hidden_states, 2)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_host, self.fc_host)

        elif 'xlnet' in self.model_type:
            attention_mask = attention_mask.float()
            outputs = self.xlnet_model(input_ids=ids, attention_mask=attention_mask, token_type_ids=seg_ids)
            hidden_states = outputs[1]

            fuse_hidden = self.get_hidden_state_by_index(hidden_states, -1)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)

            if self.extra_token:
                fuse_hidden_catg = self.get_hidden_state_by_index(hidden_states, 1)
                logits_catg = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_category, self.fc_category)

                fuse_hidden_host = self.get_hidden_state_by_index(hidden_states, 2)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_host, self.fc_host)

        elif 'roberta' in self.model_type:
            attention_mask = attention_mask.float()
            outputs = self.roberta_model(input_ids=ids, attention_mask=attention_mask, token_type_ids=seg_ids)
            hidden_states = outputs[1]

            fuse_hidden = self.get_hidden_state_by_index(hidden_states, 0)
            logits = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1, self.fc)

            if self.extra_token:
                fuse_hidden_catg = self.get_hidden_state_by_index(hidden_states, 1)
                logits_catg = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_category, self.fc_category)

                fuse_hidden_host = self.get_hidden_state_by_index(hidden_states, 2)
                logits_host = self.get_logits_by_random_dropout(fuse_hidden, self.fc_1_host, self.fc_host)
        else:
            raise NotImplementedError
        if self.extra_token:
            return logits, logits_catg, logits_host
        else:
            return logits


def test_net(extra_token = True):
    input_ids = torch.tensor([[1,2,3,4,5,0,0],[2,3,4,7,10,0,0]])
    seg_ids = torch.tensor([[0,0,0,0,0,0,0],[1,1,1,1,1,1,1]])
    model = QuestNet(extra_token=True)

    if extra_token:
        y, y_catg, y_host = model(input_ids, seg_ids)
        print(y)
        print(y_catg)
        print(y_host)
    else:
        y = model(input_ids, seg_ids)
        print(y)
    print("test completed")
    return None

if __name__  == "__main__":
    print("-------testing without extra token---------")
    test_net(extra_token=False)
    print("-------testing without extra token---------")
    test_net(extra_token=True)








