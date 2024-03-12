import nltk
from nltk.collocations import *

#setup
file = 'sanitized_requirements_corpus.txt'
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

#first up let's spit out or freqdist of most common terms
fdist = nltk.FreqDist(tokens)
print("Most common terms:")
for word, frequency in fdist.most_common(50):
    print(f"{word}: {frequency}")

#get our bigrams
finder = BigramCollocationFinder.from_words(tokens, window_size=bigram_search_window_size)
finder.apply_freq_filter(frequency_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)

print("\nBigrams sorted by frequency:")
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



