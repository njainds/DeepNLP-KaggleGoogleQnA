# google_quest_qa
kaggle
different varieties of models
#sudo python3 train_model.py --model_name "bert-base-uncased" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
sudo python3 train_model.py --model_name "bert-base-cased" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
sudo python3 train_model.py --model_name "bert-base-uncased" --content "Question_Answer" --lr_scheduler "WarmRestart" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
sudo python3 train_model.py --model_name "bert-base-uncased" --content "Question_Answer" --lr_scheduler "WarmRestart" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 8 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
sudo python3 train_model.py --model_name "roberta-base" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
#sudo python3 train_model.py --model_type "xlnet" --model_name "xlnet-base-cased" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
#sudo python3 train_model.py --model_name "bert-large-uncased" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 2 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
#sudo python3 train_model.py --model_type "xlnet" --model_name "xlnet-large-cased" --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
sudo python3 train_model.py --model_name "bert-base-uncased" --extra_token --content "Question_Answer" --max_len 512 --fold 0 --seed 2020 --split "MultilabelStratifiedKFold" --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4

Results:
1. bert-base-uncased, batch size=4, WarmupLinearSchedule, AdamW,  0.369710.(8th epoch no improvement)
2. bert-base-cased, batch size=4, WarmupLinearSchedule, AdamW     0.364466.
3. bert-large-uncased, batch size=4, WarmupLinearSchedule, AdamW  N/A
4. roberta-base, batch size=4, WarmupLinearSchedule, AdamW       0.309682 (8th epoch no improvement)
5. xlnet-base-cased, batch size=4, WarmupLinearSchedule, AdamW   N/A
6. xlnet-large-cased, batch size=4, WarmupLinearSchedule, AdamW  N/A
7. bert-base-uncased, batch size=4, WarmRestart, AdamW, fold0/9  0.367462 (diverged after Warmup restart at epoch 6 increases LR)
8. bert-base-uncased, batch size=8, WarmRestart, AdamW, fold0/9  0.385391 (diverged after Warmup restart at epoch 6 increases LR)
9. bert-base-uncased, batch size=4, WarmupLinearSchedule, AdamW, extra_token 0.269243 (still underfit)