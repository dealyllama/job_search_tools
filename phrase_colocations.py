import nltk
from nltk.collocations import *

#setup
file = 'sanitized_requirements.corpus.txt'
frequency_filter = 3
bigram_search_window_size = 3
trigram_search_window_size = 8
fourgram_search_window_size = 8



bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
fourgram_measures = nltk.collocations.QuadgramAssocMeasures()
#bgm = nltk.collocations.BigramAssocMeasures()

f = open(file, "r", encoding='utf-8')
raw = f.read()
tokens = nltk.word_tokenize(raw)

#get our bigrams
finder = BigramCollocationFinder.from_words(tokens, window_size=bigram_search_window_size)
finder.apply_freq_filter(frequency_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)

print("Bigrams sorted by frequency:")
for bigram, score in scored:
    print(f"Bigram: {bigram} Frequency: {finder.ngram_fd[bigram]} Score: {score}")

#get our trigramss
finder = TrigramCollocationFinder.from_words(tokens, window_size=trigram_search_window_size)
finder.apply_freq_filter(frequency_filter)
scored = finder.score_ngrams(trigram_measures.raw_freq)

print("\nTrigrams sorted by frequency:")
for trigram, score in scored:
    print(f"Trigram: {trigram} Frequency: {finder.ngram_fd[trigram]}  Score: {score}")

#get our fourgrams
finder = QuadgramCollocationFinder.from_words(tokens, window_size=fourgram_search_window_size)
finder.apply_freq_filter(frequency_filter)
scored = finder.score_ngrams(fourgram_measures.raw_freq)

print("\nFourgrams sorted by frequency:")
for fourgram, score in scored:
    print(f"Fourgram: {fourgram} Frequency: {finder.ngram_fd[fourgram]}  Score: {score}")



#for k,v in finder.ngram_fd.items():
#  print(k,v)

"""finder = BigramCollocationFinder.from_words(tokens, window_size=search_window_size)
scored = finder.score_ngrams(bgm.likelihood_ratio)

finder.apply_freq_filter(frequency_filter)

print("Top 50 bigrams by likelihood ratio:")
print(finder.nbest(bigram_measures.likelihood_ratio, 50))

finder = TrigramCollocationFinder.from_words(tokens, search_window_size)
finder.apply_freq_filter(frequency_filter)

print("Top 50 trigrams by likelihood ratio:")
print(finder.nbest(trigram_measures.likelihood_ratio, 50))
"""
