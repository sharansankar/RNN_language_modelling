from __future__ import division
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
class RNN:
    def __init__(self, hidden_dim=100,bptt_truncate=4):
        self.word_dim = 0 #vocabulary size
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = 0
        self.W = 0
        self.V = 0

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
    def softmax(vals):
        val_sum = np.sum(vals)
        for y in range(len(vals)):
                vals[y] = float(vals[y]/val_sum)
        return vals
    @staticmethod
    def create_matrix(segments,dict):

        filtered_segments = [[word if word in dict else '<unknown/>' for word in segment] for segment in segments]
        dict.append('<unknown/>')

        train_x = np.asarray([[dict.index(word) for word in segment[:-1]] for segment in filtered_segments])
        train_y = np.asarray([[dict.index(word) for word in segment[1:]] for segment in filtered_segments])
        return train_x, train_y

    def random_init(self):
        self.U = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim), (self.hidden_dim, self.word_dim))
        self.V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.word_dim, self.hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))

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

        self.word_dim = len(dictionary)
        print token_segments[0]
        print xtrain[0]
        return [xtrain, ytrain]

    def forward_prop(self,x_in):
        time_steps = len(x_in)

        saved_states = np.zeros((time_steps+ 1, self.hidden_dim))
        time_step_outputs = np.zeros((time_steps,self.word_dim))

        for step in range(0,time_steps):
            #print self.U[:, x_in[step]].shape
            if step == 0:
                saved_states[step] = np.tanh(self.U[:,x_in[step]])
            else:
                saved_states[step] = np.tanh(self.U[:,x_in[step]] + self.W.dot(saved_states[step-1]))

            #print self.V.dot(saved_states[step]).shape
            time_step_outputs[step] = self.softmax(self.V.dot(saved_states[step]))
        return [time_step_outputs, saved_states]

    def predict(self,x):
        out, store = self.forward_prop(x)
        return np.argmax(out,axis=1)


    def cross_entropy_loss(self,x,y_real):
        loss = 0
        N = sum([len(y_real[a]) for a in range(len(y_real))])
        for index in range(len(y_real)):
            y_predict, saved = self.forward_prop(x[index])
            predict = y_predict[[val for val in range(len(y_real[index]))], y_real[index]] #gives the probability predicted for each word that was the actual outcome
            #in this case each real outcome yt has val -1

            loss += -1 * np.nansum(np.log(predict))
        loss = loss/N
        return loss

if __name__ == '__main__':
    x = RNN()
    input, input_y = x.preprocess('corpora/shakespeare_sonnets.txt')
    x.random_init()
    #x.forward_prop(input[0])
    # r = np.random.randn(3,3)
    # print r
    # print x.softmax(r)

    print x.cross_entropy_loss(input[:1000],input_y[:1000])
