import os, nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')    

#setup
file = 'job_requirements_corpus.txt'
output_file = 'sanitized_requirements_corpus.txt'
file_text = open(file, "r", encoding='utf-8').read()

#do some initial cleanup
clean_text = re.sub(f"[{re.escape(punctuation)}]", " ", file_text)

#make everything lower cases
clean_text = clean_text.lower()

#TODO - should probably move these to a stopwords list
#remove all instances of e g
clean_text = re.sub(r"e g ", " ", clean_text)
#remove all instances of experience - it doens't seem to bring any value to the analysis
clean_text = re.sub(r"experience", "", clean_text)
#remove all instances of related - it doens't seem to bring any value to the analysis
clean_text = re.sub(r"related", "", clean_text)

stop_words = set(stopwords.words('english'))

#get our sentences
sentences = sent_tokenize(clean_text)

#remove stop words
for i in range(len(sentences)):
    sentences[i] = sentences[i].lower()
    words = word_tokenize(sentences[i])
    clean_sentence = [word for word in words if word not in stop_words]
    sentences[i] = ' '.join(clean_sentence)

#Lemmatize the words
lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    words = word_tokenize(sentences[i])
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    sentences[i] = ' '.join(lemmatized_words)


#pass an iterable in to FreqDist to get the frequency of the words in sentences
frequency_distribution = FreqDist(word_tokenize(' '.join(sentences)))
print("Top 50 words by frequency:")
print(frequency_distribution.most_common(50))

#write the sanitized corpus to a file
print("outputting sanitized corpus to file " + output_file)
with open(output_file, "w", encoding='utf-8') as file:
    file.write(' '.join(sentences))



    
        

