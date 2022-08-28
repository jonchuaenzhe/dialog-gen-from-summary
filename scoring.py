#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File: /data07/binwang/research/abs_sum_ctrl_len/rouge_s.py
# Project: /data07/binwang/research/abs_sum_ctrl_len
# Created Date: 2022-01-05 13:21:47
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import logging

def bleu_scores(reference_list, candidate_list):
    '''
        load and display scores
    '''
    scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []} 
    for i in range(len(reference_list)):
        reference = [reference_list[i].split()]
        candidate = candidate_list[i].split()
        scores["bleu_1"].append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
        scores["bleu_2"].append(sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
        scores["bleu_3"].append(sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
        scores["bleu_4"].append(sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
    
    bleu = []
    bleu.append(sum(scores["bleu_1"])/len(scores["bleu_1"]))
    bleu.append(sum(scores["bleu_2"])/len(scores["bleu_2"]))
    bleu.append(sum(scores["bleu_3"])/len(scores["bleu_3"]))
    bleu.append(sum(scores["bleu_4"])/len(scores["bleu_4"]))
    
    logging.info('\t{}: {:5.3f}\t{}: {:5.3f}\t{}: {:5.3f}\t{}: {:5.3f}'.format('bleu-1', bleu[0], 'bleu-2', bleu[1], 'bleu-3', bleu[2], 'bleu-4', bleu[3]))
    
    return bleu



def meteor_scores(reference_list, candidate_list):
    '''
        load and display scores
    '''
    scores = []
    for i in range(len(reference_list)):
        reference = [reference_list[i].split()]
        candidate = candidate_list[i].split()
        scores.append(meteor_score(reference, candidate))
    
    meteor = sum(scores)/len(scores)
    
    logging.info('\t{}: {:5.3f}'.format('meteor', meteor))
    
    return meteor



