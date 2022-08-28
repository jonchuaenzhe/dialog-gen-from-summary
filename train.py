#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: 2021-12-03 10:23:45
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
# 2022-03-31 12:49:55	B.W	re-organize and delete experimental settings
# 2022-01-25 10:54:58	B.W	compatability for label smoothing and sim loss
# 2022-01-17 16:42:06	B.W	label smoothening added and works
# 2022-01-10 14:26:43	B.W	bug fixing for loading model with 'accelerator'
# 2022-01-10 10:12:13	B.W	add testing during training to test the discripancy between val/test
# 2022-01-09 19:06:20	B.W	organize
###

import math
import os
import pprint
import logging

import nltk
import numpy as np
import torch
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import AdamW, get_scheduler, set_seed

from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

from args import parse_args
from data_loader import raw_data_loader, data_processor
from model_loader import model_loader
from rouge_s import py_rouge_scores
from scoring import bleu_scores, meteor_scores
from utils import label_smoothed_nll_loss, postprocess_text


# =  =  =  =  =  =  =  =  =  = Logging Setup =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# =  =  =  =  =  =  =  =  =  = Pre-check Package Info =  =  =  =  =  =  =  =  =  =  =  = 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# = = = = = = = = = = = = = Main Process = = = = = = = = = = = = = = = = = =
def main():
    args = parse_args()
    
    # Display Parameters
    logging.info("*** Parameters ***")
    for item, value in vars(args).items():
        logging.info("{}: {}".format(item, value))
    logging.info("")

    # Initialize the accelerator. The accelerator will handle device placement for us.
    accelerator = Accelerator()
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        #datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        #datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load raw dataset
    raw_datasets = raw_data_loader(args)

    # load model (config, tokenizer, s2s model)
    config, tokenizer, model = model_loader(accelerator, logger, args)
    
    # data processor (for DataLoader)
    dataloader, processed_dataset = data_processor(logger, args, accelerator, raw_datasets, tokenizer, model)
    train_dataloader, eval_dataloader, test_dataloader = dataloader
    train_dataset, _, _ = processed_dataset

    # = = = Training Preparation = = =
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]

    no_decay_emb_matrix = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_emb_matrix)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


    # = = = = = = = = = = = = = = = = Train = = = = = = = = = = = = = = = = = = =
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), desc="Training: ", disable=not accelerator.is_local_main_process)
    completed_steps = 0

    val_results = []
    acc_losses  = []
    best_r2_f1  = None
    best_epoch  = 0
    
    if args.model_type == 'bart' or args.model_type == 't5':
        task_specific_params = model.config.task_specific_params
        params = task_specific_params.get('summarization', {})
        params['min_length'] = args.min_target_length
        params['max_length'] = args.max_target_length
        params['length_penalty'] = args.length_penalty
        model.config.update(params)
    else:
        raise ValueError('{} model type not implemented'.format(args.model_type))


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = Train =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):

            # original model with and without label smoothing implementation
            if args.label_smoothing == 0:
                outputs = model(**batch)
                loss = outputs.loss
            else:
                outputs = model(**batch)
                output_logits = outputs.logits
                output_probs = torch.nn.functional.log_softmax(output_logits, dim=-1)
                output_probs = output_probs.view(-1, model.config.vocab_size)

                gt_logits = batch['labels']
                gt_logits = gt_logits.view(-1)

                loss, _ = label_smoothed_nll_loss(output_probs, gt_logits, args.label_smoothing, ignore_index=tokenizer.pad_token_id)

            acc_losses.append(loss.item())
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(lr=lr_scheduler.get_last_lr()[0], loss=np.mean(acc_losses[-50:]))
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = EVAL =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
        model.eval()
        val_predict     = []
        val_groundtruth = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                val_predict.extend(decoded_preds)
                val_groundtruth.extend(decoded_labels)

        logger.info("")
        logger.info("Rouge score on val set after epoch {}".format(epoch+1))
        eval_results = py_rouge_scores(val_predict, val_groundtruth)
        bleu = bleu_scores(val_groundtruth, val_predict)
        meteor = meteor_scores(val_groundtruth, val_predict)
        val_results.append(val_results)

        if best_r2_f1 is None:
            best_r2_f1 = eval_results

        if eval_results['rouge-2']['f'] >= best_r2_f1['rouge-2']['f']:
            best_r2_f1 = eval_results
            best_epoch = epoch + 1

            os.makedirs(args.output_dir+'/best', exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir+'/best', save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir+'/best')

            # Save Vocab
            vocab = tokenizer.vocab.copy()
            vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
            with open(args.output_dir + '/best/vocab.txt', 'w') as f:
                for word, index in vocab.items():
                    # it lead to encoding bug on some machines, so i add this line
                    word = word.encode('ascii', 'ignore').decode('ascii')
                    f.write(str(index) + ': ' + word + '\n')

        logger.info("Current Best Validation Result is at epoch {}".format(best_epoch))
        py_rouge_scores(None, None, best_r2_f1)



    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = Test =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
    # Load Best Model
    logger.info("Loading Best Result is at epoch {} for Testing".format(best_epoch))

    unwrapped_model = accelerator.unwrap_model(model)
    config          = config.from_pretrained(args.output_dir+'/best')
    tokenizer       = tokenizer.from_pretrained(args.output_dir+'/best', config=config)
    unwrapped_model = unwrapped_model.from_pretrained(args.output_dir+'/best', config=config)
    model           = accelerator.prepare(unwrapped_model)

    # Start Testing
    logger.info("Collecting Testing Result...")
    model.eval()

    test_predict     = []
    test_groundtruth = []

    for step, batch in enumerate(tqdm(test_dataloader, leave=False)):
        with torch.no_grad():
            
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds  = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            decoded_preds  = [' '.join(sent.split('\n')) for sent in decoded_preds]
            decoded_labels = [' '.join(sent.split('\n')) for sent in decoded_labels]

            test_predict.extend(decoded_preds)
            test_groundtruth.extend(decoded_labels)

    logger.info("")
    logger.info("Rouge score on test set")
    test_scores = py_rouge_scores(test_predict, test_groundtruth)
    bleu = bleu_scores(test_groundtruth, test_predict)
    meteor = meteor_scores(test_groundtruth, test_predict)


    # Save Train / Val / Test Scores
    with open(args.output_dir + '/scores.txt', 'w') as f:
        for item in val_results:
            f.write('\n Validation Scores:\n')
            f.write(pprint.pformat(item, indent=4))
            f.write('\n')
        f.write('\n Testing Scores:\n')
        f.write(pprint.pformat(test_scores, indent=4))

    # Save Generated Summary
    os.makedirs(args.output_dir + '/gen_samples', exist_ok=True)
    for i in range(len(test_predict)):
        test_id        = raw_datasets['test']['id'][i]
        test_dialogue  = raw_datasets['test']['dialogue'][i]
        test_summary   = raw_datasets['test']['summary'][i]
        test_predict_s = test_predict[i]

        with open(args.output_dir+'/gen_samples/'+str(test_id)+'.txt', 'w') as f:
            test_dialogue = test_dialogue.encode('ascii', 'ignore').decode('ascii')
            f.write(test_dialogue)
            f.write('\n\n')
            f.write('Golden Summary:\n')
            test_summary = test_summary.encode('ascii', 'ignore').decode('ascii')
            f.write(test_summary)
            f.write('\n\n')
            f.write('Generate Summary:\n')
            test_predict_s = test_predict_s.encode('ascii', 'ignore').decode('ascii')
            f.write(test_predict_s)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Main Process
if __name__ == "__main__":
    main()
