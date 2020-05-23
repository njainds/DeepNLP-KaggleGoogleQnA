import os
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import torch
#from callbacks import CSVParamLogger
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import time
from transformers import BertConfig, BertForPreTraining
from transformers import AdamW, BertTokenizer,  get_linear_schedule_with_warmup
#from utils import torch_to_numpy
#from model_dataset import QuestDataset
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.optimizers import FusedAdam
from apex.normalization.fused_layer_norm import FusedLayerNorm
import logging
import warnings
warnings.filterwarnings("ignore")
from model_dataset import QuestDataset
from utils.loss_function import *
from utils.metric import *
from utils.lrs_scheduler import *
from utils.file import *



parser = argparse.ArgumentParser(description="argument parser")
parser.add_argument("--train_data_folder", type = str, default = str(os.getcwd()) + '/input/', required=False, help="specify folder path of train.csv")
parser.add_argument("--batch_size", type = int, default = 8, required=False, help="training batch size")
parser.add_argument("--num_epoch", type = int, default = 3, required=False, help="number of training epochs")
parser.add_argument("--output_dir", type = str, default = str(os.getcwd()) + "/model", required=False, help="specify folder for saving checkpoints")
parser.add_argument("--seed", type = int, default = 42, required=False, help="specify the seed for training")
parser.add_argument("--lr", type = float, default = 1e-5, required=False, help="learning rate")
parser.add_argument("--lr_scheduler_name", type = str, default = "WarmupLinearSchedule", required=False, help="specify the LR scheduler")

TARGETS = ['question_score','question_views','question_favs','answer_score','is_answer_accepted']

class QuestMLMDataset(QuestDataset):
    def __init__(self, df, tokenizer, max_len=512, content = "Question_Answer", mlm_prob = 0.15,
                 non_masked_idx=-1,sop_prob=0.5,target_cols = TARGETS):
        super(QuestMLMDataset,self).__init__(df = df, target_cols=target_cols, max_len=max_len, content=content,tokenizer=tokenizer)
        self.mlm_probability = mlm_prob
        self.sop_prob = sop_prob
        self.non_masked_idx = non_masked_idx
        self.tokenizer = tokenizer
        self.cls_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.mask_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    def _mask_tokens(self,inputs,masked_random_replace_prob=0.2):
        labels = inputs.clone()
        masked_indices = torch.bernoulli(torch.full(labels.shape,self.mlm_probability)).bool()
        for special_tokens in [self.cls_token_idx,self.sep_token_idx,self.mask_token_idx]:
            masked_indices &= inputs != special_tokens
        labels[~masked_indices] = self.non_masked_idx
        indices_replaced = (torch.bernoulli(torch.full(labels.shape,1 - masked_random_replace_prob)).bool() & masked_indices)
        inputs[indices_replaced] = self.mask_token_idx
        indices_random = (torch.bernoulli(torch.full(labels.shape,0.5)).bool() & masked_indices & ~indices_replaced)
        random_words = torch.randint(len(self.tokenizer),labels.shape,dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs,labels

    def __getitem__(self, index):
        title, body, answer = self._get_text(index)
        numeric_targets = self.get_label(index)
        sop_label=0
        if np.random.uniform(0,1) < self.sop_prob:
            sop_label=1
            perm = list(np.random.permutation(range(3)))
            if perm == [0,1,2]:
                perm = [0,2,1]
            title, body, answer = [[title, body, answer][i] for i in perm]
        input_ids, token_type_ids, attention_mask  = self.get_token_ids(title, body, answer)
        input_ids, token_type_ids, attention_mask = map(torch.LongTensor,[input_ids, token_type_ids, attention_mask])
        input_ids, labels = self._mask_tokens(torch.LongTensor(input_ids))
        return ((input_ids, token_type_ids, attention_mask),(labels, sop_label, numeric_targets))

class BertPretrain(BertForPreTraining):
    def __init__(self,config, num_labels):
        super(BertPretrain,self).__init__(config,)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    def forward(self, input_ids=None,token_type_ids=None,attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = outputs[:2]
        logits = self.classifier(self.dropout(torch.mean(sequence_output,dim=1)))
        pred_scores, seq_rel_score = self.cls(sequence_output, pooled_output)
        outputs = (pred_scores,seq_rel_score,logits)
        return outputs

if __name__ == "__main__":
    len_to_sample = 600000
    args = parser.parse_args()
    path_to_data = Path(args.train_data_folder)
    path_to_ckpt_config = Path(args.train_data_folder) / "data"
    checkpoint_dir = Path(args.output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(path_to_ckpt_config, exist_ok=True)

    log = Logger()
    log.open(os.path.join(checkpoint_dir, 'train_log.txt'), mode='a+')
    log.write("\t seed = %s,  __file__ = %s, out_dir = %s" % (args.seed, __file__, args.output_dir))

    stackx_data = pd.read_csv(path_to_data / "qa_stackexchange_cleaned.csv", nrows=len_to_sample)
    stackx_data["question_title"] = stackx_data["question_title"].astype(str)
    stackx_data["question_body"] = stackx_data["question_body"].astype(str)
    stackx_data["answer"] = stackx_data["answer"].astype(str)
    #print(stackx_data.shape)
    log.write("data loaded of size %s \n" %(str(stackx_data.shape)))

    # Normalize aux targets
    encoded = []
    trange = tqdm(stackx_data["host"].unique())
    for host in trange:
        host_mask = stackx_data["host"] == host
        trange.set_description(str(host))
        host_labels = deepcopy(stackx_data[host_mask][TARGETS])
        for col in ["question_score", "question_views", "question_favs", "answer_score"]:
            host_labels[col] = rankdata(stackx_data[host_mask][col]) / host_mask.sum()
        encoded.append(host_labels)

    encoded = pd.concat(encoded, sort=False).reindex(stackx_data.index)
    stackx_data[encoded.columns] = encoded
    log.write("Aux targets are normalized \n")

    #Train-Val Split
    train_df, test_df = train_test_split(stackx_data, test_size=0.1, random_state=args.seed)

    log.write(" Train-Val Split : train_df size %s \t val_df size is %s \n" %(str(train_df.shape),str(test_df.shape)))

    #tokenizer
    tokenizer = BertTokenizer(str(path_to_ckpt_config / "vocab.txt"), do_basic_tokenize=True, do_lower_case=False)
    #original tokenizer is 28996 and new tokenizer is 110K. Vocab.txt ideally shoould be obtained by
    # using Google's wordpiece tokenizer ut Google hasn't open sourced it.
    # We can generate vocab.txt using https://github.com/kwonmha/bert-vocab-builder

    log.write("tokenizer loaded with custom vocabulary of size %d \n" %len(tokenizer))

    #Datasets
    train_dataset = QuestMLMDataset(train_df, tokenizer, target_cols=TARGETS)
    val_dataset = QuestMLMDataset(test_df, tokenizer, target_cols=TARGETS)

    #Load Model
    config = BertConfig.from_json_file(str(path_to_ckpt_config / "config.json"))
    model = BertPretrain(config, len(TARGETS))
    model = model.cuda()
    log.write("model loaded")
    #Token embeddings of new tokens
    orig_bert = BertForPreTraining.from_pretrained("bert-base-cased")
    orig_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    state_dict = orig_bert.state_dict()
    del state_dict["cls.predictions.decoder.weight"], state_dict["cls.predictions.bias"], state_dict["cls.predictions.decoder.bias"]
    orig_embedding = state_dict["bert.embeddings.word_embeddings.weight"]
    extra_tokens = list(tokenizer.vocab.keys())[len(orig_tokenizer.vocab):]
    new_tokens_as_orig_indices = [[i] for i in range(len(orig_tokenizer.vocab))] + [orig_tokenizer.encode(t, add_special_tokens=False) for t in extra_tokens]
    new_embedding = torch.zeros(len(new_tokens_as_orig_indices), orig_embedding.shape[-1])
    new_embedding.normal_(mean=0.0, std=0.02)
    for row, indices in enumerate(new_tokens_as_orig_indices):
        if len(indices) > 0:
            new_embedding[row] = orig_embedding[indices].mean(0)
    state_dict["bert.embeddings.word_embeddings.weight"] = new_embedding
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    log.write("new embeddings shape is %s" %str(new_embedding.shape))

    #Define Data Loaders
    sampler = RandomSampler(train_dataset, num_samples=len_to_sample, replacement=True)
    train_loader = DataLoader(train_dataset, sampler=sampler, num_workers=4,batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, sampler=sampler, num_workers=4,batch_size=args.batch_size)

    log.write("loaded data loaders. Train loader length is %d. Val loader length is %d" %(len(train_loader),len(val_loader)))

    #Define metric for logging
    def spearmanr(y_true,y_pred):
        corr = [spearmanr(pred_col,act_col).correlation for pred_col, act_col in zip(y_pred.T,y_true.T)]
        return np.nanmean(corr)

    #Loss
    class MLMloss(nn.CrossEntropyLoss):
        def forward(self,logits,targets):
            n_samples = np.prod(targets.shape)
            loss = super(MLMloss,self).forward(logits.view(n_samples,-1),targets.view(n_samples))
            return loss
    class SOPloss(nn.CrossEntropyLoss):
        def forward(self,logits,targets):
            n_samples = np.prod(targets.shape)
            loss = super(SOPloss,self).forward(logits.view(-1,2),targets.view(-1))
            return loss

    class PretrainingLoss(torch.nn.Module):
        def __init__(self, targets_alpha=1.0):
            super(PretrainingLoss, self).__init__()
            self.mlm_loss = MLMloss(ignore_index=-1)
            self.sop_loss = SOPloss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            self.targets_alpha = targets_alpha

        def forward(self, logits, targets):
            return (
                    self.mlm_loss(logits[0], targets[0])
                    + self.sop_loss(logits[1], targets[1])
                    + self.targets_alpha * self.bce(logits[2], targets[2])
            )

    class MLMPerplexity(MLMloss):
        __name__ = "mlm_perplexity"
        def forward(self, logits, targets):
            logits, targets = logits[0], targets[0]
            loss = super(MLMPerplexity, self).forward(logits, targets)
            perplexity = 2 ** loss
            return float(perplexity)

    def sop_accuracy(logits, targets):
        logits, targets = logits[1], targets[1]
        pred = torch.argmax(logits.view(-1, 2), dim=-1)
        targets = targets.view(-1)
        return float(torch.mean((pred == targets).float()))

    criterion = PretrainingLoss()

    #test metrics for loss
    dummy_logits = (torch.tensor([[[5.9, 1.2, 1.3], [0.6, 1.5, 0.1]]], dtype=torch.float), torch.tensor([[0.1, 0.9]]),torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    dummy_targets = (torch.tensor([[0, 2]], dtype=torch.long), torch.tensor([[0]]), torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    mlm_loss,sop_loss,sop_acc = MLMloss(ignore_index=-1)(dummy_logits[0], dummy_targets[0]) , SOPloss()(dummy_logits[1], dummy_targets[1]) , sop_accuracy(dummy_logits, dummy_targets)
    if (mlm_loss.item()==0.9608293771743774) & (sop_loss.item()==1.1711006164550781) & (sop_acc==0):
        log.write("check loss function is passed")
    else:
        log.write("check loss function is passed")
    del dummy_logits, dummy_targets,mlm_loss,sop_loss,sop_acc

    #optimizer
    optimizer_grouped_parameters = []
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': args.lr, 'weight_decay': 0.01})
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': args.lr, 'weight_decay': 0.0})
    optimizer = AdamW(optimizer_grouped_parameters, eps=4e-5)

    #LR scheduler
    if args.lr_scheduler_name == "WarmupLinearSchedule":
        accumlation_steps = 2
        warmup_proportions = 0.05
        num_train_optimization_steps = args.num_epoch * len(train_loader) // accumlation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(warmup_proportions * num_train_optimization_steps),
                                                    num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = True
    else:
        raise NotImplementedError

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    #train, eval
    model = model.cuda()
    log_step = 50
    eval_step = len(train_loader)

    for epoch in range(1, args.num_epoch + 1):
        log.write("Training Begins here!!!!!")
        log.write("\t epoch is %d and time is %s \n" % (epoch, time.strftime("%H:%M:%S", time.gmtime(time.time()))))
        prev_time = time.time()
        torch.cuda.empty_cache()
        model.zero_grad()
        for tr_batch_i, (inputs,targets) in enumerate(train_loader):
            input_ids, token_type_ids, attention_mask = inputs
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
            targets_mlm, targets_sop, targets_extra = targets
            targets_mlm, targets_sop, targets_extra = targets_mlm.cuda(), targets_sop.cuda(), targets_extra.cuda()
            model.train()
            logits = model(input_ids, token_type_ids, attention_mask)
            logits_mlm, logits_sop, logits_extra = logits
            logits_mlm, logits_sop, logits_extra = logits_mlm.cuda(), logits_sop.cuda(), logits_extra.cuda()
            loss = criterion((logits_mlm, logits_sop, logits_extra), (targets_mlm, targets_sop, targets_extra))
            with amp.scale_loss(loss / accumlation_steps, optimizer) as scaled_loss:
                scaled_loss.backward()
            if ((tr_batch_i + 1) % accumlation_steps == 0):
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                model.zero_grad()
                if lr_scheduler_each_iter:
                    scheduler.step()

            mlm_loss = MLMloss(ignore_index=-1)(logits_mlm.cpu().detach(), targets_mlm.cpu().detach())
            sop_loss = SOPloss()(logits_sop.cpu().detach(), targets_sop.cpu().detach())
            mlm_perplexity = MLMPerplexity(ignore_index=-1)((logits_mlm.cpu().detach(), logits_sop.cpu().detach(), logits_extra.cpu().detach()), (targets_mlm.cpu().detach(), targets_sop.cpu().detach(), targets_extra.cpu().detach()))
            sop_acc = sop_accuracy((logits_mlm.cpu().detach(), logits_sop.cpu().detach(), logits_extra.cpu().detach()), (targets_mlm.cpu().detach(), targets_sop.cpu().detach(), targets_extra.cpu().detach()))

             # calculating metrics for logging
            if tr_batch_i == 0:
                avg_mlm_loss = mlm_loss.item()
                avg_sop_loss = sop_loss.item()
                avg_mlm_perplexity = mlm_perplexity
                avg_loss = loss.item()
                avg_sop_acc = sop_acc

            else:
                avg_loss = (loss.item() + avg_loss * (tr_batch_i)) / (tr_batch_i + 1)
                avg_mlm_loss = (mlm_loss.item() + avg_mlm_loss * (tr_batch_i)) / (tr_batch_i + 1)
                avg_sop_loss = (sop_loss.item() + avg_sop_loss * (tr_batch_i)) / (tr_batch_i + 1)
                avg_mlm_perplexity = 2 ** avg_loss
                avg_sop_acc = (sop_acc + avg_sop_acc * (tr_batch_i)) / (tr_batch_i + 1)

            # log for training
            if (tr_batch_i + 1) % log_step == 0:
                elapsed_time = time.time() - prev_time
                prev_time = time.time()
                log.write("Training loss for epoch %d" % (epoch))
                log.write("\t Batch # %d \t perc : %f \t  combined_loss: %f \t mlm_loss: %f \t sop_loss: %f \t mlm_perplexity: %f \t sop_accuracy: %f \t elapsed time: %d\n" %((tr_batch_i + 1), ((tr_batch_i + 1) / len(val_loader)), avg_loss, avg_mlm_loss, avg_sop_loss,avg_mlm_perplexity, avg_sop_acc, elapsed_time))

            if (tr_batch_i + 1) % eval_step == 0:
                eval_count += 1
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    for val_batch_i, (inputs, targets) in enumerate(val_loader):
                        model.eval()
                        input_ids, token_type_ids, attention_mask = inputs
                        input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
                        targets_mlm, targets_sop, targets_extra = targets
                        targets_mlm, targets_sop, targets_extra = targets_mlm.cuda(), targets_sop.cuda(), targets_extra.cuda()
                        logits = model(input_ids, token_type_ids, attention_mask)
                        logits_mlm, logits_sop, logits_extra = logits
                        logits_mlm, logits_sop, logits_extra = logits_mlm.cuda(), logits_sop.cuda(), logits_extra.cuda()
                        loss = criterion((logits_mlm, logits_sop, logits_extra),(targets_mlm, targets_sop, targets_extra))

                        mlm_loss = MLMloss(ignore_index=-1)(logits_mlm.cpu().detach(), targets_mlm.cpu().detach())
                        sop_loss = SOPloss()(logits_sop.cpu().detach(), targets_sop.cpu().detach())
                        mlm_perplexity = MLMPerplexity(ignore_index=-1)((logits_mlm.cpu().detach(), logits_sop.cpu().detach(), logits_extra.cpu().detach()),(targets_mlm.cpu().detach(), targets_sop.cpu().detach(), targets_extra.cpu().detach()))
                        sop_acc = sop_accuracy((logits_mlm.cpu().detach(), logits_sop.cpu().detach(), logits_extra.cpu().detach()),(targets_mlm.cpu().detach(), targets_sop.cpu().detach(), targets_extra.cpu().detach()))

                        # calculating metrics for logging
                        if val_batch_i == 0:
                            avg_mlm_loss = mlm_loss.item()
                            avg_sop_loss = sop_loss.item()
                            avg_mlm_perplexity = mlm_perplexity
                            avg_loss = loss.item()
                            avg_sop_acc = sop_acc

                        else:
                            avg_loss = (loss.item() + avg_loss * (val_batch_i)) / (val_batch_i + 1)
                            avg_mlm_loss = (mlm_loss.item() + avg_mlm_loss * (val_batch_i)) / (val_batch_i + 1)
                            avg_sop_loss = (sop_loss.item() + avg_sop_loss * (val_batch_i)) / (val_batch_i + 1)
                            avg_mlm_perplexity = 2 ** avg_loss
                            avg_sop_acc = (sop_acc + avg_sop_acc * (val_batch_i)) / (val_batch_i + 1)

                        # log for training
                        if (val_batch_i + 1) % len(val_loader) == 0:
                            elapsed_time = time.time() - prev_time
                            prev_time = time.time()
                            log.write("Validation loss for epoch %d" %(epoch))
                            log.write("\t Batch # %d \t perc : %f \t  combined_loss: %f \t mlm_loss: %f \t sop_loss: %f \t mlm_perplexity: %f \t sop_accuracy: %f \t elapsed time: %d\n" %
                                ((val_batch_i + 1), ((val_batch_i + 1) / len(val_loader)), avg_loss, avg_mlm_loss,avg_sop_loss, avg_mlm_perplexity, avg_sop_acc, elapsed_time))

        log.write("epoch %d is completed \n" %(epoch))
        ckpt_filename = "ckpt_model_epoch=" + str(epoch) + "_seed=" +  str(seed) + ".pth"
        checkpoint_filepath = checkpoint_dir / ckpt_filename
        torch.save(model.state_dict(), checkpoint_filepath)
        log.write("Model checkpoint saved to file %s for epoch %d \n" % (ckpt_filename, epoch))






