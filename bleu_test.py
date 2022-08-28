import nltk

from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
print(sentence_bleu(reference, candidate))


reference = [['Hi', 'Tom,', 'are', 'you', 'busy', 'tomorrow’s', 'afternoon?']]
candidate = ['Hi', 'Tom', 'are', 'you', 'busy', "tomorrow", 'afternoon?']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print(score)