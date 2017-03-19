from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
class RNN:
    def __init__(self):
        self.xtrain = 0
        self.ytrain = 0
        return

    @staticmethod
    def tokenize_and_append(segments):
        for x in range(len(segments)):
            segments[x] = WhitespaceTokenizer().tokenize(segments[x])
            segments[x].append('\n') #only for shakespeare sonets

            #append start and end symbols to each sentence
            segments[x] = ['<s>'] + segments[x]
            segments[x].append('</s>')
        return segments


    @staticmethod
    def create_matrix(segments,dict):

        filtered_segments = [[word if word in dict else '<unknown/>' for word in segment] for segment in segments]
        dict.append('<unknown/>')

        train_x = np.asarray([[dict.index(word) for word in segment[:-1]] for segment in filtered_segments])
        train_y = np.asarray([[dict.index(word) for word in segment[1:]] for segment in filtered_segments])
        return train_x, train_y

    def preprocess(self, infile):
        input = open(infile,'r')
        segments = []
        token_segments = []

        for line in input:
            if len(line) > 1:
                segments.append(line)
                token_segments.append(line)

        #tokenize string
        token_segments = self.tokenize_and_append(token_segments)
        segments = self.tokenize_and_append(segments)

        for x in range(len(segments)):
            segments[x] = ' '.join(segments[x])

        #get mapping of each word
        vectorizer = TfidfVectorizer(min_df=2,tokenizer=lambda x:WhitespaceTokenizer().tokenize(x))
        vectorizer.fit_transform(segments)
        dictionary = vectorizer.get_feature_names()

        xtrain, ytrain = self.create_matrix(token_segments,dictionary)

        self.xtrain = xtrain
        self.ytrain = ytrain

        print token_segments[0]
        print xtrain[0]


if __name__ == '__main__':
    x = RNN()
    x.preprocess('corpora/shakespeare_sonnets.txt')
