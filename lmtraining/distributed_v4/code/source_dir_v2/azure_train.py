#https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/distributed-pytorch-with-horovod/distributed-pytorch-with-horovod.ipynb
#https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/training/distributed-pytorch-with-horovod/pytorch_horovod_mnist.py
#https://horovod.readthedocs.io/en/latest/_modules/horovod/torch.html
import os

from copy import deepcopy
from pathlib import Path
import numpy as np
import random
import pandas as pd
import argparse
import torch
# from callbacks import CSVParamLogger
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import time
from transformers import BertConfig, BertForPreTraining
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
# from apex.optimizers import FusedAdam
# from apex.normalization.fused_layer_norm import FusedLayerNorm
import logging
import warnings
import horovod.torch as hvd

warnings.filterwarnings("ignore")
from model_dataset import QuestDataset
from utils.loss_function import *
from utils.metric import *
from utils.lrs_scheduler import *
from utils.file import *

from azureml.core.run import Run

run = Run.get_context()

parser = argparse.ArgumentParser(description="argument parser")
parser.add_argument("--train_data_folder", type=str, default=str(os.getcwd()) + '/input/', required=False,
                    help="specify folder path of train.csv")
parser.add_argument("--batch_size", type=int, default=8, required=False, help="training batch size")
parser.add_argument("--num_epoch", type=int, default=3, required=False, help="number of training epochs")
parser.add_argument("--output_dir", type=str, default=str(os.getcwd()) + "/model", required=False,
                    help="specify folder for saving checkpoints")
parser.add_argument("--seed", type=int, default=42, required=False, help="specify the seed for training")
parser.add_argument("--lr", type=float, default=1e-5, required=False, help="learning rate")
parser.add_argument("--lr_scheduler_name", type=str, default="WarmupLinearSchedule", required=False,
                    help="specify the LR scheduler")
parser.add_argument("--path_to_ckpt_config", type=str, default=str(os.getcwd()) + '/input/data/', required=False,
                    help="specify folder path of train.csv")

TARGETS = ['question_score', 'question_views', 'question_favs', 'answer_score', 'is_answer_accepted']


class QuestMLMDataset(QuestDataset):
    def __init__(self, df, tokenizer, max_len=128, content="Question_Answer", mlm_prob=0.15,
                 non_masked_idx=-1, sop_prob=0.5, target_cols=TARGETS):
        super(QuestMLMDataset, self).__init__(df=df, target_cols=target_cols, max_len=max_len, content=content,
                                              tokenizer=tokenizer)
        self.mlm_probability = mlm_prob
        self.sop_prob = sop_prob
        self.non_masked_idx = non_masked_idx
        self.tokenizer = tokenizer
        self.cls_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.mask_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    def _mask_tokens(self, inputs, masked_random_replace_prob=0.2):
        labels = inputs.clone()
        masked_indices = torch.bernoulli(torch.full(labels.shape, self.mlm_probability)).bool()
        for special_tokens in [self.cls_token_idx, self.sep_token_idx, self.mask_token_idx]:
            masked_indices &= inputs != special_tokens
        labels[~masked_indices] = self.non_masked_idx
        indices_replaced = (
                    torch.bernoulli(torch.full(labels.shape, 1 - masked_random_replace_prob)).bool() & masked_indices)
        inputs[indices_replaced] = self.mask_token_idx
        indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __getitem__(self, index):
        title, body, answer = self._get_text(index)
        numeric_targets = self.get_label(index)
        sop_label = 0
        if np.random.uniform(0, 1) < self.sop_prob:
            sop_label = 1
            perm = list(np.random.permutation(range(3)))
            if perm == [0, 1, 2]:
                perm = [0, 2, 1]
            title, body, answer = [[title, body, answer][i] for i in perm]
        input_ids, token_type_ids, attention_mask = self.get_token_ids(title, body, answer)
        input_ids, token_type_ids, attention_mask = map(torch.LongTensor, [input_ids, token_type_ids, attention_mask])
        input_ids, labels = self._mask_tokens(torch.LongTensor(input_ids))
        return ((input_ids, token_type_ids, attention_mask), (labels, sop_label, numeric_targets))


class BertPretrain(BertForPreTraining):
    def __init__(self, config, num_labels):
        super(BertPretrain, self).__init__(config, )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = outputs[:2]
        logits = self.classifier(self.dropout(torch.mean(sequence_output, dim=1)))
        pred_scores, seq_rel_score = self.cls(sequence_output, pooled_output)
        outputs = (pred_scores, seq_rel_score, logits)
        return outputs


def spearmanr(y_true, y_pred):
    corr = [spearmanr(pred_col, act_col).correlation for pred_col, act_col in zip(y_pred.T, y_true.T)]
    return np.nanmean(corr)


# Loss
class MLMloss(nn.CrossEntropyLoss):
    def forward(self, logits, targets):
        n_samples = np.prod(targets.shape)
        loss = super(MLMloss, self).forward(logits.view(n_samples, -1), targets.view(n_samples))
        return loss


class SOPloss(nn.CrossEntropyLoss):
    def forward(self, logits, targets):
        n_samples = np.prod(targets.shape)
        loss = super(SOPloss, self).forward(logits.view(-1, 2), targets.view(-1))
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


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda =  torch.cuda.is_available()
    hvd.init()
    torch.manual_seed(args.seed)
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    len_to_sample = 600000
    path_to_data = Path(args.train_data_folder)
    path_to_ckpt_config = Path(args.path_to_ckpt_config)
    checkpoint_dir = Path(args.output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(path_to_ckpt_config, exist_ok=True)


    def metric_average(val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    log = Logger()
    log.open(os.path.join(checkpoint_dir, 'train_log.txt'), mode='a+')
    log.write("\t seed = %s,  __file__ = %s, out_dir = %s" % (args.seed, __file__, args.output_dir))

    log.write("current rank is %s and total processes is %s" % (str(hvd.rank()),str(hvd.size())))

    accumlation_steps = 1
    # args.batch_size = int(args.batch_size / accumlation_steps)
    # batch size is actually the minibatch size sample: total batch size is 128 with 2 gpus each on 4 nodes. so batch size is 128/(2*4)=16
    random.seed(args.seed)
    np.random.seed(args.seed)

    stackx_data = pd.read_csv(path_to_data / "qa_stackexchange_cleaned.csv", nrows=len_to_sample)
    stackx_data["question_title"] = stackx_data["question_title"].astype(str)
    stackx_data["question_body"] = stackx_data["question_body"].astype(str)
    stackx_data["answer"] = stackx_data["answer"].astype(str)
    # print(stackx_data.shape)
    log.write("data loaded of size %s \n" % (str(stackx_data.shape)))

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

    # Train-Val Split
    train_df, test_df = train_test_split(stackx_data, test_size=0.1, random_state=args.seed)

    log.write(" Train-Val Split : train_df size %s \t val_df size is %s \n" % (str(train_df.shape), str(test_df.shape)))

    # tokenizer
    tokenizer = BertTokenizer(str(path_to_ckpt_config / "vocab.txt"), do_basic_tokenize=True, do_lower_case=False)
    # original tokenizer is 28996 and new tokenizer is 110K. Vocab.txt ideally shoould be obtained by
    # using Google's wordpiece tokenizer ut Google hasn't open sourced it.
    # We can generate vocab.txt using https://github.com/kwonmha/bert-vocab-builder

    log.write("tokenizer loaded with custom vocabulary of size %d \n" % len(tokenizer))

    # Datasets
    train_dataset = QuestMLMDataset(train_df, tokenizer, target_cols=TARGETS)
    val_dataset = QuestMLMDataset(test_df, tokenizer, target_cols=TARGETS)

    # Load Model
    config = BertConfig.from_json_file(str(path_to_ckpt_config / "config.json"))
    model = BertPretrain(config, len(TARGETS))
    log.write("model loaded")
    if args.cuda:
    # Move model to GPU.
        model.cuda()

    # Token embeddings of new tokens
    orig_bert = BertForPreTraining.from_pretrained("bert-base-cased")
    orig_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    state_dict = orig_bert.state_dict()
    del state_dict["cls.predictions.decoder.weight"], state_dict["cls.predictions.bias"], state_dict[
        "cls.predictions.decoder.bias"]
    orig_embedding = state_dict["bert.embeddings.word_embeddings.weight"]
    extra_tokens = list(tokenizer.vocab.keys())[len(orig_tokenizer.vocab):]
    new_tokens_as_orig_indices = [[i] for i in range(len(orig_tokenizer.vocab))] + [
        orig_tokenizer.encode(t, add_special_tokens=False) for t in extra_tokens]
    new_embedding = torch.zeros(len(new_tokens_as_orig_indices), orig_embedding.shape[-1])
    new_embedding.normal_(mean=0.0, std=0.02)
    for row, indices in enumerate(new_tokens_as_orig_indices):
        if len(indices) > 0:
            new_embedding[row] = orig_embedding[indices].mean(0)
    state_dict["bert.embeddings.word_embeddings.weight"] = new_embedding
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    log.write("new embeddings shape is %s" % str(new_embedding.shape))

    sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    v_sampler = DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size)
    #Keeping validation as non-distributed
    val_loader = DataLoader(val_dataset,sampler = v_sampler, batch_size=args.batch_size)


    num_train_optimization_steps = args.num_epoch * len(train_loader) // accumlation_steps
    #t_total = num_train_optimization_steps// hvd.size()
    t_total = num_train_optimization_steps

    criterion = PretrainingLoss()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # optimizer
    optimizer_grouped_parameters = []
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': args.lr,
         'weight_decay': 0.01})
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'lr': args.lr,
         'weight_decay': 0.0})
    #Lr should be scaled lineary with batch size sclaing for distributed training.
    #https://arxiv.org/abs/1706.02677
    optimizer = AdamW(optimizer_grouped_parameters, eps=4e-5)
    #compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=model.named_parameters(),compression=hvd.Compression.none)
    log.write("\n\n len of train sampler: %d \n" %len(sampler))
    log.write("\n\n len of train loader: %d \n" % len(train_loader))
    log.write("\n\n t_total num_train_optimization_steps : %d \n" % t_total)


    # LR scheduler
    if args.lr_scheduler_name == "WarmupLinearSchedule":
        warmup_proportions = 0.05
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_proportions * t_total),
                                                    num_training_steps=t_total)
        lr_scheduler_each_iter = True
    else:
        raise NotImplementedError
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    run.log('lr', np.float(args.lr))

    # train, eval
    # model = model.to(device)
    log_step = 100
    eval_step = len(train_loader)

    for epoch in range(1, args.num_epoch + 1):
        log.write("Training Begins here!!!!!")
        log.write("\t epoch is %d and time is %s \n" % (epoch, time.strftime("%H:%M:%S", time.gmtime(time.time()))))
        prev_time = time.time()
        sampler.set_epoch(epoch)
        model.zero_grad()
        model.train()

        for tr_batch_i, (inputs, targets) in enumerate(train_loader):
            input_ids, token_type_ids, attention_mask = inputs
            input_ids, token_type_ids, attention_mask = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda()
            targets_mlm, targets_sop, targets_extra = targets
            targets_mlm, targets_sop, targets_extra = targets_mlm.cuda(), targets_sop.cuda(), targets_extra.cuda()
            optimizer.zero_grad()
            logits = model(input_ids, token_type_ids, attention_mask)
            logits_mlm, logits_sop, logits_extra = logits
            logits_mlm, logits_sop, logits_extra = logits_mlm.cuda(), logits_sop.cuda(), logits_extra.cuda()
            loss = criterion((logits_mlm, logits_sop, logits_extra), (targets_mlm, targets_sop, targets_extra))

            # with amp.scale_loss(loss / accumulation_steps, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            if ((tr_batch_i + 1) % accumlation_steps == 0):
                #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                if lr_scheduler_each_iter:
                    scheduler.step()

            mlm_loss = MLMloss(ignore_index=-1)(logits_mlm, targets_mlm)
            sop_loss = SOPloss()(logits_sop, targets_sop)
            mlm_perplexity = MLMPerplexity(ignore_index=-1)((logits_mlm, logits_sop, logits_extra),(targets_mlm, targets_sop, targets_extra))
            sop_acc = sop_accuracy((logits_mlm, logits_sop, logits_extra),(targets_mlm, targets_sop,targets_extra))

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
                lr = 0
                for param_group in optimizer.param_groups:
                    lr += param_group['lr']/len(optimizer.param_groups)
                log.write("Training loss for epoch %d" % (epoch))
                log.write(
                    "\t Batch # %d \t perc : %f \t  combined_loss: %f \t mlm_loss: %f \t sop_loss: %f \t mlm_perplexity: %f \t sop_accuracy: %f \t  lr: %.6f \t elapsed time: %d\n" % (
                    (tr_batch_i + 1), ((tr_batch_i + 1) / len(train_loader)), avg_loss, avg_mlm_loss, avg_sop_loss,
                    avg_mlm_perplexity, avg_sop_acc, lr, elapsed_time))
                run.log("Training Batch #", tr_batch_i + 1)
                run.log("Training perc batch/gpu #", np.float((tr_batch_i + 1) / len(train_loader)))
                run.log("Training combined_loss #", avg_loss)
                run.log("Training mlm_loss #", avg_mlm_loss)
                run.log("Training sop_loss #", avg_sop_loss)
                run.log("Training mlm_perplexity #", avg_mlm_perplexity)
                run.log("Training sop_accuracy #", avg_sop_acc)

            if (tr_batch_i + 1) % eval_step == 0:
                # eval_count += 1
                avg_mlm_loss = 0
                avg_sop_loss = 0
                avg_mlm_perplexity = 0
                avg_loss = 0
                avg_sop_acc = 0

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

                        mlm_loss = MLMloss(ignore_index=-1)(logits_mlm, targets_mlm)
                        sop_loss = SOPloss()(logits_sop, targets_sop)
                        mlm_perplexity = MLMPerplexity(ignore_index=-1)((logits_mlm, logits_sop, logits_extra),(targets_mlm, targets_sop, targets_extra))
                        sop_acc = sop_accuracy((logits_mlm.cuda(), logits_sop.cuda(), logits_extra.cuda()),(targets_mlm.cuda(), targets_sop.cuda(), targets_extra.cuda()))

                        # calculating metrics for logging
                        avg_mlm_loss += mlm_loss.item()/len(val_loader)
                        avg_sop_loss += sop_loss.item()/len(val_loader)
                        avg_mlm_perplexity += mlm_perplexity/len(val_loader)
                        avg_loss += loss.item()/len(val_loader)
                        avg_sop_acc += sop_acc/len(val_loader)

                        avg_mlm_loss = metric_average(avg_mlm_loss, 'avg_mlm_loss')
                        avg_sop_loss = metric_average(avg_sop_loss, 'avg_sop_loss')
                        avg_mlm_perplexity = metric_average(avg_mlm_perplexity, 'avg_mlm_perplexity')
                        avg_loss = metric_average(avg_loss, 'avg_loss')
                        avg_sop_acc = metric_average(avg_sop_acc, 'avg_sop_acc')


                        # log for training
                        if (val_batch_i + 1) % len(val_loader) == 0:
                            elapsed_time = time.time() - prev_time
                            prev_time = time.time()
                            log.write("Validation loss for epoch %d" % (epoch))
                            log.write(
                                "\t Batch # %d \t perc : %f \t  combined_loss: %f \t mlm_loss: %f \t sop_loss: %f \t mlm_perplexity: %f \t sop_accuracy: %f \t elapsed time: %d\n" %
                                ((val_batch_i + 1), ((val_batch_i + 1) / len(val_loader)), avg_loss, avg_mlm_loss,
                                 avg_sop_loss, avg_mlm_perplexity, avg_sop_acc, elapsed_time))
                            run.log("Validation Batch #", val_batch_i + 1)
                            run.log("Validation perc batch/gpu #", np.float((val_batch_i + 1) / len(val_loader)))
                            run.log("Validation combined_loss #", avg_loss)
                            run.log("Validation mlm_loss #", avg_mlm_loss)
                            run.log("Validation sop_loss #", avg_sop_loss)
                            run.log("Validation mlm_perplexity #", avg_mlm_perplexity)
                            run.log("Validation sop_accuracy #", avg_sop_acc)

        run.log("epoch completed is ", epoch)
        ckpt_filename = "ckpt_model_epoch_" + str(epoch) + "_rank_" + str(
            int(hvd.rank())) + "_" + "_seed_" + str(args.seed) + ".pth"
        checkpoint_filepath = checkpoint_dir / ckpt_filename
        torch.save(model.state_dict(), checkpoint_filepath)
        log.write("Model checkpoint saved to file %s for epoch %d \n" % (ckpt_filename, epoch))
        run.log("Model checkpoint saved to file for epoch:", epoch)







