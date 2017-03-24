from __future__ import division
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import operator

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
        vals_soft = np.exp(vals - np.max(vals))
        return vals_soft/np.sum(vals_soft)
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
        # print token_segments[0]
        # print xtrain[0]
        return [xtrain, ytrain]

    def forward_prop(self,x_in):
        time_steps = len(x_in)

        saved_states = np.zeros((time_steps+ 1, self.hidden_dim))
        time_step_outputs = np.zeros((time_steps,self.word_dim))

        for step in range(0,time_steps):
            #print self.U[:, x_in[step]].shape

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
            print predict
            loss += -1 * np.nansum(np.log(predict))
        loss = loss/N
        return loss


    def backprop_tt(self,x_in,y_real):
        N = len(y_real)

        y_predict, saved_states = self.forward_prop(x_in)

        dl_dU = np.zeros(self.U.shape)
        dl_dV = np.zeros(self.V.shape)
        dl_dW = np.zeros(self.W.shape)

        delta_y = y_predict
        delta_y[np.arange(N),y_real] -= 1 #since y_real = 1

        for step_back in np.arange(N)[::-1]:
            print delta_y.shape
            buffer = np.transpose(saved_states[step_back])
            print buffer.shape
            print dl_dV.shape
            dl_dV += np.outer(delta_y[step_back],saved_states[step_back].transpose())

            delta_t = self.V.T.dot(delta_y[step_back]) * (1 - (saved_states[step_back] ** 2))
            for bptt_step in np.arange(max(0, step_back-self.bptt_truncate), step_back+1)[::-1]:
                dl_dW += np.outer(delta_t, saved_states[bptt_step-1])
                dl_dU[:,x_in[bptt_step]] += delta_t
                delta_t = self.V.T.dot(delta_y[step_back]) * (1 - (saved_states[bptt_step-1] ** 2))
        return [dl_dU, dl_dV, dl_dW]

    def gradient_check(self,x,y,h=0.001, error_threshold=0.01):
        backprop_gradients = self.backprop_tt(x,y)

        parameters = ['U', 'V', 'W']
        for parameter_index, parameter in enumerate(parameters):
            parameter = operator.attrgetter(parameter)(self)
            print "Performing gradient check for parameter %s with size %d." % (parameter, np.prod(parameter.shape))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])

            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.cross_entropy_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.cross_entropy_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = backprop_gradients[parameter_index][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold: #& gt > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (parameter, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (parameter)
        return
if __name__ == '__main__':
    #np.random.seed(10)
    x = RNN(hidden_dim=10, bptt_truncate=1000)
    #input, input_y = x.preprocess('corpora/shakespeare_sonnets.txt')
    x.word_dim = 100
    x.random_init()
    #x.forward_prop(input[:3])
    # r = np.random.randn(3,3)
    # print r
    # print x.softmax(r)

    #print x.cross_entropy_loss(input[:10],input_y[:1000])
    x.gradient_check([0,1,2,3], [1,2,3,4])


    # print "Actual loss: %f" % x.cross_entropy_loss(input[:10], input_y[:10])
    # print "Expected Loss for random predictions: %f" % np.log(x.word_dim)
