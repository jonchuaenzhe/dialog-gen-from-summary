Apr 2022 - Aug 2022
# Summary-Based Dialogue Generation for Data Augmentation with BART

Data for dialogue summarization is very limited as summaries have to be hand-annotated. A proposed solution is to use a generation model to produce more dialogue examples from the same summary label to increase the dataset. This project attempts to do so, with ultimately disappointing results as the augmented data does little to improve accuracy of dialogue summarization models.

With a small training dataset of 1000 dialogue-summary pairs, a BART-base model achieved the following on a test-set:
```
rouge-1:	P: 43.73	R: 45.98	F1: 43.45
rouge-2:	P: 17.13	R: 17.64	F1: 16.80
rouge-l:	P: 40.92	R: 42.67	F1: 40.88
bleu-1: 0.316	bleu-2: 0.112	bleu-3: 0.058	bleu-4: 0.030
meteor: 0.302
```
After augmenting the 1000 pairs with an additional 1000 generated dialogues (from the same summary labels), the BART-base model achieved the following on the same test-set:
```
rouge-1:	P: 43.06	R: 46.28	F1: 43.22
rouge-2:	P: 16.65	R: 17.62	F1: 16.53
rouge-l:	P: 40.38	R: 42.87	F1: 40.67
bleu-1: 0.314	bleu-2: 0.111	bleu-3: 0.056	bleu-4: 0.029
meteor: 0.305
```
As such, there is little improvement (actually, a slight deprovement) in the summarization results, invalidating the proposed solution.

Refer to this README for a summary of the implementation, or view the files for an in-depth understanding.

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
