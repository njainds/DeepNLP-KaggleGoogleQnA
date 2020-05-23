import os
os.environ["OMP_NUM_THREADS"]="1"

import gc
import pandas as pd
import numpy as np
import random
import argparse
from functools import partial
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CyclicLR, StepLR, _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert import BertAdam
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.optimizers import FusedAdam

from model_dataset import *
from utils.loss_function import *
from utils.metric import *
from utils.lrs_scheduler import *
from utils.file import *
from model import *

parser = argparse.ArgumentParser(description="argument parser")
parser.add_argument("--train_data_folder", type = str, default = str(os.getcwd()) + '/data/', required=False, help="specify folder path of train.csv")
parser.add_argument("--model_type", type = str, default = "bert", required=False, help="specify model type")
parser.add_argument("--model_name", type = str, default = "bert-base-uncased", required=False, help="specify model name")
parser.add_argument("--content", type = str, default = "Question_Answer", required=False, help="specify content")
parser.add_argument("--max_len", type = int, default = 512, required=False, help="specify length of tokens")
parser.add_argument("--hidden_layers", type = list, default = [-3,-4,-5,-6,-7], required=False, help="specify hidden layers for loss")
parser.add_argument("--optimizer", type = str, default = "AdamW", required=False, help="specify the optimizer")
parser.add_argument("--lr_scheduler", type = str, default = "WarmupLinearSchedule", required=False, help="specify the LR scheduler")
parser.add_argument("--warmup_proportion", type = float, default = 0.05, required=False, help="proportion of training for warmup of LR")
parser.add_argument("--lr", type = float, default = 3e-5, required=False, help="initial value of learning rate")
parser.add_argument("--batch_size", type = int, default = 8, required=False, help="training batch size")
parser.add_argument("--valid_batch_size", type = int, default = 32, required=False, help="validation batch size")
parser.add_argument("--num_epoch", type = int, default = 12, required=False, help="number of training epochs")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument("--num_workers", type = int, default = 2, required=False, help="# workers for testing data loader")
parser.add_argument("--start_epoch", type = int, default = 0, required=False, help="start epoch for continnuos training")
parser.add_argument("--checkpoint_folder", type = str, default = str(os.getcwd()) + "/model", required=False, help="specify folder for saving checkpoints")
parser.add_argument("--extra_token", action = "store_true", default = False, help="whether to use extra tokens for training")
parser.add_argument("--load_pretrain", action = "store_true", default = False, help="whether to load pretrain model")
parser.add_argument("--fold", type = int, default = 0, required=True, help="fold to train model")
parser.add_argument("--seed", type = int, default = 42, required=True, help="specify the seed for training")

parser.add_argument("--split", type = str, default = "GroupKFold", required = True, help="type of split for train-val")
parser.add_argument("--augment", action = "store_true", default = False, help="whether to load pretrain model")
parser.add_argument("--loss", type = str, default = "mse", required=True, help="loss for training")
parser.add_argument("--n_splits", type = int, default = 5, required=True, help="num splits for splitting data for training")
parser.add_argument("--early_stopping", type = int, default = 3, required = False, help="num epochs for early stopping ")

# Mandatory: loss,n_splits,split,fold,seed
# Optional but critical: num_epoch,batch_size,valid_batch_size,extra_token, model_type, model_name,content,hidden_layers, max_len, lr, lr_scheduler, optimizer,warmup_proportions

num_category_class = 5
num_host_class  = 64
auxiliary_weights = [1,0.05, 0.05]
decay_factor = 0.9
min_lr = 2e-6

unbalance_weight = [1,1,1,1,1,1,1,1,1,2,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
q_unbalance_weight = unbalance_weight[:21]
a_unbalance_weight = unbalance_weight[21:]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHseed"] =str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

def training(tokenizer, content, n_splits, fold, train_data_loader, val_data_loader, model_type, model_name, hidden_layers,
             optimizer_name, lr_scheduler_name, lr, warmup_proportions, batch_size, valid_batch_size, num_epoch, start_epoch, accumlation_steps,
             checkpoint_folder, load_pretrain, seed, loss, extra_token, augment, early_stopping):
    torch.cuda.empty_cache()
    strng = "@%s: \n" % os.path.basename(__file__)
    strng += "\tset random seed = %d \n" % seed
    strng += "\t cuda environment: \n"
    strng += "\t torch version is %s \t torch.version.cuda is %s \t torch.backends.cudnn.version() = %s \n" % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
    strng += "\t torch.cuda.device_count() is %s  \n" % (torch.cuda.device_count())

    if augment:
        if extra_token:
            checkpoint_folder = os.path.join(checkpoint_folder,model_type + '/' + model_name + '-' + content + '-' + loss + '-' + optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + 'aug_differential_extra_token/')
        else:
            checkpoint_folder = os.path.join(checkpoint_folder,model_type + '/' + model_name + '-' + content + '-' + loss + '-' + optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + 'aug_differential/')
    else:
        if extra_token:
            checkpoint_folder = os.path.join(checkpoint_folder,model_type + '/' + model_name + '-' + content + '-' + loss + '-' + optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + 'extra_token/')
        else:
            checkpoint_folder = os.path.join(checkpoint_folder,model_type + '/' + model_name + '-' + content + '-' + loss + '-' + optimizer_name + '-' + lr_scheduler_name + '-' + str(n_splits) + '-' + str(seed) + '-' + '/')

    checkpoint_filename = 'fold_' + str(fold) + "_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)

    os.makedirs(checkpoint_folder, exist_ok=True)

    log = Logger()
    log.open(os.path.join(checkpoint_folder, 'fold_' + str(fold) + '_train_log.txt'), mode = 'a+')
    log.write('\t%s\n' % strng)
    log.write("\t seed = %s, fold = %s, __file__ = %s, out_dir = %s" %(seed, fold, __file__, checkpoint_folder))

    def load(model, pretrain_file, skip=[]):
        pretrain_dict = torch.load(pretrain_file)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            if any( s in key for s in skip):
                continue
            else:
                state_dict[key] = pretrain_dict[key]
        model.load_state_dict(state_dict, strict = False)
        return model

    if content == "Question_Answer":
        num_class = 30
    elif content == "Question":
        num_class = 21
    elif content == "Answer":
        num_class = 9

    if model_type == "bert":
        if extra_token:
            model = QuestNet(model_type = model_name, tokenizer=tokenizer, n_classes=num_class, n_category_classes = num_category_class, n_host_classes = num_host_class, hidden_layers = hidden_layers, extra_token = True)
        else:
            model = QuestNet(model_type=model_name, tokenizer=tokenizer, n_classes=num_class,n_category_classes=num_category_class, n_host_classes=num_host_class, hidden_layers=hidden_layers, extra_token=False)
    elif model_type == "xlnet":
        if extra_token:
            model = QuestNet(model_type = model_name, tokenizer=tokenizer, n_classes=num_class, n_category_classes = num_category_class, n_host_classes = num_host_class, hidden_layers = hidden_layers, extra_token = True)
        else:
            model = QuestNet(model_type=model_name, tokenizer=tokenizer, n_classes=num_class,n_category_classes=num_category_class, n_host_classes=num_host_class, hidden_layers=hidden_layers, extra_token=False)
    else:
        raise NotImplementedError

    model = model.cuda()
    if load_pretrain:
        if content == "Answer":
            model = load(model, checkpoint_filepath, skip=['fc.weight','fc.bias'])
        else:
            model = load(model, checkpoint_filepath)

    if model_name == "t5-base":
        weight_decay = 0.9
    else:
        weight_decay = 0.01

    if (model_type =='bert') or (model_type =='xlnet'):

        optimizer_grouped_parameters = []
        list_lr = []

        if (model_name == 'bert-base-uncased') or (model_name == 'bert-base-cased'):
            list_layers = [model.bert_model.embeddings,
                           model.bert_model.encoder.layer[0],
                           model.bert_model.encoder.layer[1],
                           model.bert_model.encoder.layer[2],
                           model.bert_model.encoder.layer[3],
                           model.bert_model.encoder.layer[4],
                           model.bert_model.encoder.layer[5],
                           model.bert_model.encoder.layer[6],
                           model.bert_model.encoder.layer[7],
                           model.bert_model.encoder.layer[8],
                           model.bert_model.encoder.layer[9],
                           model.bert_model.encoder.layer[10],
                           model.bert_model.encoder.layer[11],
                           model.fc_1,
                           model.fc
                           ]

        elif (model_name == 'bert-large-uncased'):
            list_layers = [model.bert_model.embeddings,
                           model.bert_model.encoder.layer[0],
                           model.bert_model.encoder.layer[1],
                           model.bert_model.encoder.layer[2],
                           model.bert_model.encoder.layer[3],
                           model.bert_model.encoder.layer[4],
                           model.bert_model.encoder.layer[5],
                           model.bert_model.encoder.layer[6],
                           model.bert_model.encoder.layer[7],
                           model.bert_model.encoder.layer[8],
                           model.bert_model.encoder.layer[9],
                           model.bert_model.encoder.layer[10],
                           model.bert_model.encoder.layer[11],
                           model.bert_model.encoder.layer[12],
                           model.bert_model.encoder.layer[13],
                           model.bert_model.encoder.layer[14],
                           model.bert_model.encoder.layer[15],
                           model.bert_model.encoder.layer[16],
                           model.bert_model.encoder.layer[17],
                           model.bert_model.encoder.layer[18],
                           model.bert_model.encoder.layer[19],
                           model.bert_model.encoder.layer[20],
                           model.bert_model.encoder.layer[21],
                           model.bert_model.encoder.layer[22],
                           model.bert_model.encoder.layer[23],
                           model.fc_1,
                           model.fc
                           ]
        elif (model_name == "xlnet-base-cased"):

            list_layers = [model.xlnet_model.word_embedding,
                           model.xlnet_model.layer[0],
                           model.xlnet_model.layer[1],
                           model.xlnet_model.layer[2],
                           model.xlnet_model.layer[3],
                           model.xlnet_model.layer[4],
                           model.xlnet_model.layer[5],
                           model.xlnet_model.layer[6],
                           model.xlnet_model.layer[7],
                           model.xlnet_model.layer[8],
                           model.xlnet_model.layer[9],
                           model.xlnet_model.layer[10],
                           model.xlnet_model.layer[11],
                           model.fc_1,
                           model.fc
                           ]
        elif (model_name == "roberta-base"):

            list_layers = [model.roberta_model.embeddings,
                           model.roberta_model.encoder.layer[0],
                           model.roberta_model.encoder.layer[1],
                           model.roberta_model.encoder.layer[2],
                           model.roberta_model.encoder.layer[3],
                           model.roberta_model.encoder.layer[4],
                           model.roberta_model.encoder.layer[5],
                           model.roberta_model.encoder.layer[6],
                           model.roberta_model.encoder.layer[7],
                           model.roberta_model.encoder.layer[8],
                           model.roberta_model.encoder.layer[9],
                           model.roberta_model.encoder.layer[10],
                           model.roberta_model.encoder.layer[11],
                           model.fc_1,
                           model.fc
                           ]
        elif (model_name == "gpt2"):
            list_layers = [  # model.gpt2_model.wte,
                # model.gpt2_model.wpe,
                model.gpt2_model.h[0],
                model.gpt2_model.h[1],
                model.gpt2_model.h[2],
                model.gpt2_model.h[3],
                model.gpt2_model.h[4],
                model.gpt2_model.h[5],
                model.gpt2_model.h[6],
                model.gpt2_model.h[7],
                model.gpt2_model.h[8],
                model.gpt2_model.h[9],
                model.gpt2_model.h[10],
                model.gpt2_model.h[11],
                model.fc_1,
                model.fc]
        else:
            raise NotImplementedError

        ######## Differential LR and optimizer group ############################################################

        if model_name == "":
            for layer in list_layers:
                list_lr.append(lr)
                lr = lr * decay_factor
            list_lr.reverse()
        else:
            mult = lr/min_lr
            step = mult**(1/(len(list_layers)-1))
            list_lr = [lr * (step**i) for i in range(len(list_layers))]
        no_decay = ['bias','LayerNorm.weight','LayerNorm.bias']

        for i in range(len(list_lr)):
            if isinstance(list_lr[i], list):
                for list_layer in list_layers[i]:
                    layer_parameters = list(list_layer.named_parameters())
                    optimizer_grouped_parameters.append({'params':[p for n,p in layer_parameters if not any(nd in n for nd in no_decay)],
                                                         'lr': list_lr[i],'weight_decay': weight_decay})
                    optimizer_grouped_parameters.append(
                        {'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)],
                         'lr': list_lr[i],'weight_decay': 0.0})
            else:
                layer_parameters = list(list_layers[i].named_parameters())
                optimizer_grouped_parameters.append(
                    {'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)],
                     'lr': list_lr[i],'weight_decay': weight_decay})
                optimizer_grouped_parameters.append(
                    {'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)],
                     'lr': list_lr[i],'weight_decay': 0.0})
        if extra_token:
            layer_parameters = list(model.fc_1_category.named_parameters())
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)],'lr': 1e-6, 'weight_decay': weight_decay})
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)],'lr': 1e-6, 'weight_decay': 0.0})

            layer_parameters = list(model.fc_1_host.named_parameters())
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], 'lr': 1e-6,'weight_decay': weight_decay})
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], 'lr': 1e-6,'weight_decay': 0.0})

            layer_parameters = list(model.fc_category.named_parameters())
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], 'lr': 1e-6,'weight_decay': weight_decay})
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], 'lr': 1e-6,'weight_decay': 0.0})

            layer_parameters = list(model.fc_host.named_parameters())
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if not any(nd in n for nd in no_decay)], 'lr': 1e-6,'weight_decay': weight_decay})
            optimizer_grouped_parameters.append({'params': [p for n, p in layer_parameters if any(nd in n for nd in no_decay)], 'lr': 1e-6,'weight_decay': 0.0})
        else:
            print("no extra token")
    else:
        raise NotImplementedError

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    elif optimizer_name =='Ranger':
        optimizer = Ranger(optimizer_grouped_parameters)
    elif optimizer_name =='BertAdam':
        num_optimization_steps = num_epoch * len(train_data_loader)//accumlation_steps
        optimizer = BertAdam(optimizer_grouped_parameters, warmup=warmup_proportions, t_total=num_optimization_steps)
    elif optimizer_name == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters,eps=4e-5)
    elif optimizer_name == 'FusedAdam':
        optimizer = FusedAdam(optimizer_grouped_parameters,bias_correction = False)
    else:
        raise NotImplementedError

    ######## LR shceduler ############################################################
    if lr_scheduler_name =='CosineAnealing':
        num_train_optimization_steps = num_epoch * len(train_data_loader)//accumlation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= int(warmup_proportions*num_train_optimization_steps), num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = False
    elif lr_scheduler_name =="WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)
        lr_scheduler_each_iter = False
    elif lr_scheduler_name=="WarmupLinearSchedule":
        num_train_optimization_steps = num_epoch * len(train_data_loader) // accumlation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= int(warmup_proportions*num_train_optimization_steps), num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = True
    else:
        raise NotImplementedError

    log.write("\t model name: %s \n" % model_name)
    log.write("\t optimizer name: %s \n" % optimizer_name)
    log.write("\t scheduler name: %s \n" % lr_scheduler_name)

    # AMP -automatic mixed precision training for faster training
    # https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html

    model, optimizer = amp.initialize(model, optimizer, opt_level = 'O1')
    eval_step = len(train_data_loader)
    log_step = 50
    eval_count=0
    count=0

    log.write('\t training starts here!!\n')
    log.write('\t batch size = %d, accumulation steps = %d \n' % (batch_size, accumlation_steps))
    log.write('\t experiment : %s' % str(__file__.split('/')[-2:]))

    valid_loss = np.zeros(1, np.float32)
    train_loss = np.zeros(1, np.float32)
    valid_metric_optimal = -np.inf

    writer = SummaryWriter()

    # Define loss
    if loss =='mse':
        criterion = MSELoss()
    elif loss =='mse-bce':
        criterion = MSBCELoss()
    elif loss =='focal':
        criterion = FocalLoss()
    elif loss == 'bce':
        if content == 'Question_Answer':
            weights = torch.tensor(np.array(unbalance_weight), dtype=torch.float64).cuda()
        elif content == 'Answer':
            weights = torch.tensor(np.array(a_unbalance_weight), dtype=torch.float64).cuda()
        elif content == 'Question':
            weights = torch.tensor(np.array(q_unbalance_weight), dtype=torch.float64).cuda()
        else:
            raise NotImplementedError
        criterion = nn.BCEWithLogitsLoss(weight = weights)
        criterion_extra = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    for epoch in range(1, num_epoch+1):
        labels_train = None
        pred_train = None
        labels_val = None
        pred_val = None

        if (epoch>1) and (not lr_scheduler_each_iter):
            scheduler.step()
        if epoch<start_epoch:
            continue
        log.write("\t epoch is %d and time is %s \n" %(epoch,time.strftime("%H:%M:%S", time.gmtime(time.time()))))
        prev_time = time.time()

        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        torch.cuda.empty_cache()
        model.zero_grad()

        if extra_token:
            for tr_batch_i, (token_ids, seg_ids, labels, labels_category, labels_host) in enumerate(train_data_loader):
                rate=0
                for param_group in optimizer.param_groups:
                    rate += param_group['lr']/len(optimizer.param_groups)

                model.train()
                token_ids = token_ids.cuda()
                seg_ids = seg_ids.cuda()
                labels = labels.cuda().float()
                labels_category = labels_category.cuda().float()
                labels_host = labels_host.cuda().float()

                prediction, prediction_category, prediction_host = model(token_ids, seg_ids)
                loss = auxiliary_weights[0] * criterion(prediction, labels) + auxiliary_weights[1] * criterion_extra(prediction_category, labels_category) + auxiliary_weights[2] * criterion_extra(prediction_host, labels_host)
                with amp.scale_loss(loss/accumlation_steps, optimizer) as scaled_loss:
                    scaled_loss.backward()

                if ((tr_batch_i + 1) % accumlation_steps == 0):
                    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0, norm_type=2)
                    optimizer.step()
                    model.zero_grad()
                    if lr_scheduler_each_iter:
                        scheduler.step()
                        #Write to tensorboard summary writer
                    writer.add_scalar("train_loss_" + str(fold), loss.item(), (epoch-1)*len(train_data_loader)*batch_size + tr_batch_i*batch_size)
                prediction = torch.sigmoid(prediction)
                if tr_batch_i==0:
                    pred_train = prediction.cpu().detach().numpy()
                    labels_train = labels.cpu().detach().numpy()
                else:
                    pred_train = np.concatenate((pred_train, prediction.cpu().detach().numpy()), axis=0)
                    labels_train = np.concatenate((labels_train, labels.cpu().detach().numpy()), axis=0)
                l = np.array([loss.item()*batch_size])
                n = np.array([batch_size])
                sum_train_loss += l
                sum_train += n

                #log for training
                if (tr_batch_i+1) % log_step == 0:
                    train_loss = sum_train_loss/(sum_train + 1e-12)
                    pred_train = np.nan_to_num(pred_train)
                    sp = Spearman(labels_train, pred_train)
                    elapsed_time = time.time() - prev_time
                    prev_time = time.time() 
                    log.write("\t Batch # %d \t perc processed in epoch: %f \t  train_loss is %f \t lr is %f \t spearman is %f \t elapsed time: %d\n" % ((tr_batch_i+1),((tr_batch_i+1)/len(train_data_loader)),train_loss[0], rate, sp, elapsed_time))

                if (tr_batch_i + 1) % eval_step == 0:
                    eval_count +=1
                    valid_loss = np.zeros(1, np.float32)
                    valid_num = np.zeros_like(valid_loss)

                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        for val_batch_i, (token_ids, seg_ids, labels, labels_category, labels_host) in enumerate(val_data_loader):
                            model.eval()
                            token_ids = token_ids.cuda()
                            seg_ids = seg_ids.cuda()
                            labels = labels.cuda().float()
                            labels_category = labels_category.cuda().float()
                            labels_host = labels_host.cuda().float()

                            prediction, prediction_category, prediction_host = model(token_ids, seg_ids)
                            val_loss = auxiliary_weights[0] * criterion(prediction, labels) + auxiliary_weights[1] * criterion_extra(prediction_category, labels_category) + auxiliary_weights[2] * criterion_extra(prediction_host, labels_host)
                            writer.add_scalar("val_loss_" + str(fold), val_loss.item(), (eval_count - 1) * len(val_data_loader) * valid_batch_size + val_batch_i * valid_batch_size)

                            prediction = torch.sigmoid(prediction)
                            if val_batch_i == 0:
                                pred_val = prediction.cpu().detach().numpy()
                                labels_val = labels.cpu().detach().numpy()
                            else:
                                pred_val = np.concatenate((pred_val, prediction.cpu().detach().numpy()), axis=0)
                                labels_val = np.concatenate((labels_val, labels.cpu().detach().numpy()), axis=0)
                            l = np.array([val_loss.item() * valid_batch_size])
                            n = np.array([valid_batch_size])
                            valid_loss += l
                            valid_num += n

                            valid_loss = valid_loss / (valid_num + 1e-12)
                            pred_val = np.nan_to_num(pred_val)
                            sp = Spearman(labels_val, pred_val)
                            log.write("\t Batch # %d perc processed in epoch: %f Validation loss is %f \t spearman is %f \n" % (val_batch_i,(val_batch_i/len(val_data_loader)),valid_loss[0], sp))
        else:
            for tr_batch_i, (token_ids, seg_ids, labels) in enumerate(train_data_loader):
                rate=0
                for param_group in optimizer.param_groups:
                    rate += param_group['lr']/len(optimizer.param_groups)

                model.train()
                token_ids = token_ids.cuda()
                seg_ids = seg_ids.cuda()
                labels = labels.cuda().float()

                prediction = model(token_ids, seg_ids)
                loss = criterion(prediction, labels)
                with amp.scale_loss(loss/accumlation_steps, optimizer) as scaled_loss:
                    scaled_loss.backward()

                if ((tr_batch_i + 1) % accumlation_steps == 0):
                    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0, norm_type=2)
                    optimizer.step()
                    model.zero_grad()
                    if lr_scheduler_each_iter:
                        scheduler.step()
                        #Write to tensorboard summary writer
                    writer.add_scalar("train_loss_" + str(fold), loss.item(), (epoch-1)*len(train_data_loader)*batch_size + tr_batch_i*batch_size)
                prediction = torch.sigmoid(prediction)
                if tr_batch_i==0:
                    pred_train = prediction.cpu().detach().numpy()
                    labels_train = labels.cpu().detach().numpy()
                else:
                    pred_train = np.concatenate((pred_train, prediction.cpu().detach().numpy()), axis=0)
                    labels_train = np.concatenate((labels_train, labels.cpu().detach().numpy()), axis=0)
                l = np.array([loss.item()*batch_size])
                n = np.array([batch_size])
                sum_train_loss += l
                sum_train += n

                #log for training
                if (tr_batch_i+1) % log_step == 0:
                    train_loss = sum_train_loss/(sum_train + 1e-12)
                    pred_train = np.nan_to_num(pred_train)
                    sp = Spearman(labels_train, pred_train)
                    elapsed_time = time.time() - prev_time
                    prev_time = time.time()
                    log.write("\t Batch # %d \t perc processed in epoch: %f \t  train_loss is %f \t lr is %f \t spearman is %f \t elapsed time: %d\n" % ((tr_batch_i+1),((tr_batch_i+1)/len(train_data_loader)),train_loss[0], rate, sp, elapsed_time))

                if (tr_batch_i + 1) % eval_step == 0:
                    eval_count +=1
                    valid_loss = np.zeros(1, np.float32)
                    valid_num = np.zeros_like(valid_loss)

                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        for val_batch_i, (token_ids, seg_ids, labels) in enumerate(val_data_loader):
                            model.eval()
                            token_ids = token_ids.cuda()
                            seg_ids = seg_ids.cuda()
                            labels = labels.cuda().float()
                            prediction = model(token_ids, seg_ids)
                            val_loss = criterion(prediction, labels)
                            writer.add_scalar("val_loss_" + str(fold), val_loss.item(), (eval_count - 1) * len(val_data_loader) * valid_batch_size + val_batch_i * valid_batch_size)

                            prediction = torch.sigmoid(prediction)
                            if val_batch_i == 0:
                                pred_val = prediction.cpu().detach().numpy()
                                labels_val = labels.cpu().detach().numpy()
                            else:
                                pred_val = np.concatenate((pred_val, prediction.cpu().detach().numpy()), axis=0)
                                labels_val = np.concatenate((labels_val, labels.cpu().detach().numpy()), axis=0)
                            l = np.array([val_loss.item() * valid_batch_size])
                            n = np.array([valid_batch_size])
                            valid_loss += l
                            valid_num += n

                            valid_loss = valid_loss / (valid_num + 1e-12)
                            pred_val = np.nan_to_num(pred_val)
                            sp = Spearman(labels_val, pred_val)
                            log.write("\t Batch # %d perc processed in epoch: %f \t Validation loss is %f \t spearman is %f \n" % (val_batch_i,(val_batch_i/len(val_data_loader)),valid_loss[0], sp))
        val_metric_epoch = sp
        if (val_metric_epoch > valid_metric_optimal):
            log.write("\t valid metric improved in epoch %d from %f to %f. Saving model.. \n" % (epoch,valid_metric_optimal,val_metric_epoch))
            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)
            np.savez_compressed(checkpoint_folder + '/prob_pred_fold_' + str(fold) + 'uint8.npz', pred_val)
            np.savez_compressed(checkpoint_folder + '/prob_label_fold_' + str(fold) + 'uint8.npz', labels_val)
            count = 0
        else:
            count+=1
        if count==early_stopping:
            log.write("\t early stopped as validatio  metric did not improve \n")
            break
if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.seed)
    data_path = args.train_data_folder + 'train.csv'
    get_train_val_split(data_path = data_path, save_path = args.train_data_folder, n_splits = args.n_splits, seed = args.seed, split = args.split)
    train_data_path = args.train_data_folder + "split/train_fold_%s_seed_%s.csv" % (args.fold, args.seed)
    val_data_path = args.train_data_folder + "split/val_fold_%s_seed_%s.csv" % (args.fold, args.seed)
    if ((args.model_type == "bert") or (args.model_type == "xlnet") or (args.model_type == "t5")):

        if args.extra_token:
            test_data_path = args.train_data_folder + "test.csv"
            train_df = pd.read_csv(data_path)
            test_df = pd.read_csv(test_data_path)

            train_host_list = train_df['host'].unique().tolist()
            test_host_list = test_df['host'].unique().tolist()
            host_encoder = LabelBinarizer()
            host_encoder.fit(list(set(train_host_list + test_host_list)))

            train_category_list = train_df['category'].unique().tolist()
            test_category_list = test_df['category'].unique().tolist()
            category_encoder = LabelBinarizer()
            category_encoder.fit(list(set(train_category_list + test_category_list)))

            train_data_loader, val_data_loader, tokenizer = get_train_val_loaders(train_data_path=train_data_path,
                                                                                  val_data_path=val_data_path,
                                                                                  host_encoder=host_encoder,
                                                                                  category_encoder=category_encoder,
                                                                                  max_len=args.max_len,
                                                                                  model_type=args.model_name,
                                                                                  content=args.content,
                                                                                  batch_size=args.batch_size,
                                                                                  val_batch_size=args.valid_batch_size,
                                                                                  num_workers=args.num_workers,
                                                                                  augment=args.augment,
                                                                                  extra_token=True)

        else:
            train_data_loader, val_data_loader, tokenizer = get_train_val_loaders(train_data_path=train_data_path,
                                                                                  val_data_path=val_data_path,
                                                                                  model_type=args.model_name,
                                                                                  content=args.content,
                                                                                  max_len=args.max_len,
                                                                                  batch_size=args.batch_size,
                                                                                  val_batch_size=args.valid_batch_size,
                                                                                  num_workers=args.num_workers,
                                                                                  augment=args.augment,
                                                                                  extra_token=False)
    else:
        raise NotImplementedError

        # start training
    training(tokenizer, args.content, args.n_splits, args.fold, train_data_loader, val_data_loader, args.model_type, args.model_name, args.hidden_layers, args.optimizer, args.lr_scheduler, args.lr, args.warmup_proportion, args.batch_size, args.valid_batch_size, args.num_epoch, args.start_epoch, args.accumulation_steps, args.checkpoint_folder, args.load_pretrain, args.seed, args.loss, args.extra_token, args.augment, args.early_stopping)
    gc.collect()















































