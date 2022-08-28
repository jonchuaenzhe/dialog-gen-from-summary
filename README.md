Apr 2022 - Aug 2022
# Summary-Based Dialogue Generation for Data Augmentation with BART

Data for dialogue summarization is very limited as summaries have to be hand-annotated. A proposed solution is to use a generation model to produce more dialogue examples from the same summary label to increase the dataset. This project attempts to do so, with ultimately disappointing results as the augmented data does little to improve accuracy of dialogue summarization models.

With a small training set of 1000 dialogue-summary pairs, a BART model achieved the following on a test-set:
```
rouge-1:	P: 43.73	R: 45.98	F1: 43.45
rouge-2:	P: 17.13	R: 17.64	F1: 16.80
rouge-l:	P: 40.92	R: 42.67	F1: 40.88
bleu-1: 0.316	bleu-2: 0.112	bleu-3: 0.058	bleu-4: 0.030	meteor: 0.302
```
After augmenting the 1000 pairs with an additional 1000 generated dialogues (from the same summary labels), the same BART model achieved the following on the same test-set:
```
rouge-1:	P: 43.06	R: 46.28	F1: 43.22
rouge-2:	P: 16.65	R: 17.62	F1: 16.53
rouge-l:	P: 40.38	R: 42.87	F1: 40.67
bleu-1: 0.314	bleu-2: 0.111	bleu-3: 0.056	bleu-4: 0.029	meteor: 0.305
```
As seen, there is negligible change in the summarization results, invalidating the proposed solution. Refer to this README for a summary of the implementation, or view the files for an in-depth understanding.

## Set-up

Install the following dependencies
```
conda install -c anaconda nltk
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install transformers==4.13.0
pip install accelerate==0.5.1
conda install -c conda-forge datasets=1.15.1
conda install -c anaconda scipy
pip install py-rouge
```

## Dialog Generation Model

The SAMSUM dataset was used for training a dialogue generation model. A dialogue and its corresponding summary was fed into the model, with one utterance being masked. With additional info such as "utterance length" and "overlap with summary", the model was trained to generate the utterance. An example input is as follows:
```
Summary - Rashi is confused by too many career choices. Teacher advises him to choose something he has passion for and what interests him.
Dialogue - 
Teacher: Rashi, why are you so low? 
<MASK>
Teacher: What is your confusion?
Rashi: I was discussing with my friends about the career options. 
Teacher: Hmm.
Rashi: There are too many to choose from.
Teacher: Choose a career based on what truly interests you. 

Speaker - Rashi
Overlap - 1, Total - 22
Length - M
```
The training label is as such:
```
Ma’am I’m a bit confused about my career.
```
The model performed decently in predicting the masked utterances, with the following results:
```
rouge-1:	P: 29.56	R: 26.81	F1: 27.44
rouge-2:	P: 13.66	R: 12.42	F1: 12.73
rouge-l:	P: 31.37	R: 28.94	F1: 29.59
bleu-1: 0.182	bleu-2: 0.079	bleu-3: 0.045	bleu-4: 0.026	meteor: 0.168
```

## Finetuning the Generation Model




























