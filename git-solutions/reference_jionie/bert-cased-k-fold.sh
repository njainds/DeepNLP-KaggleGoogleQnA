# 10 fold bert-base-cased, question_answer, seed 1996
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 0 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 1 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 2 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 3 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 4 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 5 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 6 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 7 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 8 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 9 --seed 1996 --n_splits 10 --batch_size 4 --valid_batch_size 32 --accumulation_steps 2 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4


# 5 fold bert-base-cased, question + answer, seed 1996
python training-k-fold.py --model_name "bert-base-cased" --content "Answer" --max_len 512 --fold 0 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4 
python training-k-fold.py --model_name "bert-base-cased" --content "Answer" --max_len 512 --fold 1 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4 
python training-k-fold.py --model_name "bert-base-cased" --content "Answer" --max_len 512 --fold 2 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4 
python training-k-fold.py --model_name "bert-base-cased" --content "Answer" --max_len 512 --fold 3 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4 
python training-k-fold.py --model_name "bert-base-cased" --content "Answer" --max_len 512 --fold 4 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4 

python training-k-fold.py --model_name "bert-base-cased" --content "Question" --max_len 512 --fold 0 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_name "bert-base-cased" --content "Question" --max_len 512 --fold 1 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_name "bert-base-cased" --content "Question" --max_len 512 --fold 2 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_name "bert-base-cased" --content "Question" --max_len 512 --fold 3 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4
python training-k-fold.py --model_name "bert-base-cased" --content "Question" --max_len 512 --fold 4 --seed 1996 --n_splits 5 --split "GroupKfold" --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --loss "bce" --augment --num_epoch 8 --num_workers 4


# new version bert-cased for Ivan, 5 fold bert-base-cased, question_answer, seed 726 
# python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 0 --seed 726 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --split "GroupKfold" --loss "bce" --augment --num_epoch 8 --num_workers 4
# python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 1 --seed 726 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --split "GroupKfold" --loss "bce" --augment --num_epoch 8 --num_workers 4
# python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 2 --seed 726 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --split "GroupKfold" --loss "bce" --augment --num_epoch 8 --num_workers 4
# python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 3 --seed 726 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --split "GroupKfold" --loss "bce" --augment --num_epoch 8 --num_workers 4
# python training-k-fold.py --model_type "bert" --content "Question_Answer" --model_name "bert-base-cased" --fold 4 --seed 726 --n_splits 5 --batch_size 8 --valid_batch_size 32 --accumulation_steps 1 --lr 1e-4 --split "GroupKfold" --loss "bce" --augment --num_epoch 8 --num_workers 4
