#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/CTRLen/data_loader.py
# Project: /data07/binwang/research/CTRLen
# Created Date: 2021-12-06 11:10:03
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
# 2022-02-12 13:49:10	B.W	implement data shuffle
############


# import lib
import json
import csv
import random
import logging
import numpy as np

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq


def raw_data_loader(args):
    ''' load raw datasets from csv files'''

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    if 'samsum' in args.train_file:
        train_dict      = load_mask_info_from_samsum(args, args.train_file)
        validation_dict = load_mask_info_from_samsum(args, args.validation_file)
        test_dict       = load_mask_info_from_samsum(args, args.test_file)

        raw_datasets = datasets.DatasetDict({"train":train_dict, "validation":validation_dict, "test":test_dict})

    elif 'dialogsum' in args.train_file:
        train_dict      = load_from_dialogsum(args, args.train_file)
        validation_dict = load_from_dialogsum(args, args.validation_file)
        test_dict       = load_from_dialogsum(args, args.test_file)

        raw_datasets = datasets.DatasetDict({"train":train_dict, "validation":validation_dict, "test":test_dict})

    if args.shuffle:
        logging.info("shuffle the dataset")
        raw_datasets = data_shuffle(raw_datasets)

    return raw_datasets


# Masking Functions
def string_overlap(summary_list, utterance_list):
    count = 0
    
    for word in utterance_list:
        if word in summary_list:
            count += 1
    
    return count

def length_bucket(utterance_list):
    length = len(utterance_list)
    
    if length <= 4:
        return "S"
    if length > 10:
        return "L"
    
    return "M"
# Masking Functions


def load_mask_info_from_samsum(args, file_path):
    ''' load samsum csv data '''

    id_list       = []
    dialogue_list = []
    summary_list  = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            Id = row['id']
            dialogue = row['dialogue']
            summary = row['summary']

            dialogue_sep = dialogue.split('\n')
            length = len(dialogue_sep)

            input_str = "Summary - " + summary + "\n" + "Dialogue - \n"
            for i in range(length):
                try:
                    speaker = dialogue_sep[i].split(':')[0]
                    target = dialogue_sep[i].split(':')[1]

                    speaker = "Speaker - " + speaker + "\n"

                    overlap = string_overlap(summary.split(), target.split())
                    total = len(summary.split())
                    add_info = "Overlap - " + str(overlap) + ", Total - " + str(total) + "\n"

                    length_info = "Length - " + length_bucket(target.split())

                    temp_dialogue = dialogue_sep.copy()
                    temp_dialogue[i] = "<mask>"
                    temp_dialogue = '\n'.join(temp_dialogue)

                    id_list.append(Id + '_' + str(i))
                    dialogue_list.append(input_str + temp_dialogue + '\n\n' + speaker + add_info + length_info)
                    summary_list.append(target)
                except:
                    pass

    data_dict = {'id': id_list,
                 'dialogue': dialogue_list,
                 'summary': summary_list
                 }

    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def load_from_samsum(args, file_path):
    ''' load samsum csv data '''

    id_list       = []
    dialogue_list = []
    summary_list  = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id_list.append(row['id'])
            dialogue_list.append("Summary - " + row['summary'] + "\n" + "Dialogue - ")
            summary_list.append(row['dialogue'])

    data_dict = {'id': id_list,
                 'dialogue': dialogue_list,
                 'summary': summary_list
                 }

    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def load_mask_info_from_dialogsum(args, file_path):
    ''' load samsum csv data '''

    data = []

    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    id_list       = [sample['fname'] for sample in data]
    dialogue_list = [sample['dialogue'] for sample in data]

    if 'summary' in data[0]:
        summary_list  = [sample['summary'] for sample in data]

    elif 'summary1' in data[0]:

        id_list1 = [id+"_sum1" for id in id_list]
        id_list2 = [id+"_sum2" for id in id_list]
        id_list3 = [id+"_sum3" for id in id_list]

        id_list = id_list1 + id_list2 + id_list3
        dialogue_list = dialogue_list + dialogue_list + dialogue_list

        summary_list1  = [sample['summary1'] for sample in data]
        summary_list2  = [sample['summary2'] for sample in data]
        summary_list3  = [sample['summary3'] for sample in data]

        summary_list = summary_list1 + summary_list2 + summary_list3

    new_id_list       = []
    new_dialogue_list = []
    new_summary_list  = []
    list_len = len(summary_list)

    for entry in range(list_len):
        Id = id_list[entry]
        dialogue = dialogue_list[entry]
        summary = summary_list[entry]

        dialogue_sep = dialogue.split('\n')
        length = len(dialogue_sep)

        input_str = "Summary - " + summary + "\n" + "Dialogue - \n"
        for i in range(length):
            try:
                speaker = dialogue_sep[i].split(':')[0]
                target = dialogue_sep[i].split(':')[1]

                speaker = "Speaker - " + speaker + "\n"

                overlap = string_overlap(summary.split(), target.split())
                total = len(summary.split())
                add_info = "Overlap - " + str(overlap) + ", Total - " + str(total) + "\n"

                length_info = "Length - " + length_bucket(target.split())

                temp_dialogue = dialogue_sep.copy()
                temp_dialogue[i] = "<mask>"
                temp_dialogue = '\n'.join(temp_dialogue)

                new_id_list.append(Id + '_' + str(i))
                new_dialogue_list.append(input_str + temp_dialogue + '\n\n' + speaker + add_info + length_info)
                new_summary_list.append(target)
            except:
                pass

    data_dict = {'id': new_id_list,
                 'dialogue': new_dialogue_list,
                 'summary': new_summary_list
                 }

    data_dict = Dataset.from_dict(data_dict)

    return data_dict



def load_from_dialogsum(args, file_path):
    ''' load dialogue jsonl data '''

    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    id_list       = [sample['fname'] for sample in data]
    dialogue_list = [sample['dialogue'] for sample in data]

    if 'summary' in data[0]:
        summary_list  = [sample['summary'] for sample in data]

    elif 'summary1' in data[0]:

        id_list1 = [id+"_sum1" for id in id_list]
        id_list2 = [id+"_sum2" for id in id_list]
        id_list3 = [id+"_sum3" for id in id_list]

        id_list = id_list1 + id_list2 + id_list3
        dialogue_list = dialogue_list + dialogue_list + dialogue_list

        summary_list1  = [sample['summary1'] for sample in data]
        summary_list2  = [sample['summary2'] for sample in data]
        summary_list3  = [sample['summary3'] for sample in data]

        summary_list = summary_list1 + summary_list2 + summary_list3


    data_dict = {'id': id_list,
                'dialogue': dialogue_list,
                'summary': summary_list}

    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def data_shuffle(raw_datasets):
    ''' shuffle the dataset '''

    num_train = len(raw_datasets['train'])
    num_val   = len(raw_datasets['validation'])
    num_test  = len(raw_datasets['test'])

    id_list       = raw_datasets['train']['id'] + raw_datasets['validation']['id'] + raw_datasets['test']['id']
    dialogue_list = raw_datasets['train']['dialogue'] + raw_datasets['validation']['dialogue'] + raw_datasets['test']['dialogue']
    summary_list  = raw_datasets['train']['summary'] + raw_datasets['validation']['summary'] + raw_datasets['test']['summary']

    all_data = list(zip(id_list, dialogue_list, summary_list))
    random.shuffle(all_data)

    id_list, dialogue_list, summary_list = zip(*all_data)

    # train
    train_dict = { 
        'id'      : id_list[:num_train],
        'dialogue': dialogue_list[:num_train],
        'summary' : summary_list[:num_train]
        }
    train_dict = Dataset.from_dict(train_dict)

    # validation
    val_dict = { 
        'id'      : id_list[num_train:num_train+num_val],
        'dialogue': dialogue_list[num_train:num_train+num_val],
        'summary' : summary_list[num_train:num_train+num_val]
        }
    val_dict = Dataset.from_dict(val_dict)

    # test
    test_dict = { 
        'id'      : id_list[num_train+num_val:num_train+num_val+num_test],
        'dialogue': dialogue_list[num_train+num_val:num_train+num_val+num_test],
        'summary' : summary_list[num_train+num_val:num_train+num_val+num_test]
        }
    test_dict = Dataset.from_dict(test_dict)

    raw_datasets = datasets.DatasetDict({"train":train_dict, "validation":val_dict, "test":test_dict})

    return raw_datasets


def data_processor(logger, args, accelerator, raw_datasets, tokenizer, model):
    ''' prepare dataset format for train/val/test '''

    def preprocess_function(examples):

        # summary - target
        targets = examples[summary_column]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # dialogue - input
        inputs = examples[text_column]
        new_inputs = []
        for i, inp in enumerate(inputs):
            new_inputs.append(prefix + inp)

        inputs = new_inputs
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    summary_column = args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

    return (train_dataloader, eval_dataloader, test_dataloader), (train_dataset, eval_dataset, test_dataset)


