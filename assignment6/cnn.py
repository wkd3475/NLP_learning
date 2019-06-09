import torch
import torch.nn
import random
import pickle
import time

DIMENTION = 300
NUM_FILTERS = 100
NUM_CLASSES = 2
TRAIN_BATCH_SIZE = 50
NUM_FILTER_FEATURE_MAPS = 100
EPOCH = 1

class Sentence():
    sentence = None
    words = None
    state = None

    def __init__(self, sentence, state):
        self.sentence = sentence
        self.state = state
        
    def set_words(self, words):
        self.words = words

    def print_information(self):
        print(self.sentence)
        print(self.words)
        print(self.state)

def preprocessing():
    word_list = []
    sentence_list = []
    w2i = {}

    pos_list = open('rt-polarity.pos', mode='r', encoding='latin-1').read().splitlines()
    neg_list = open('rt-polarity.neg', mode='r', encoding='latin-1').read().splitlines()

    for line in pos_list:
        translation_table = dict.fromkeys(map(ord, '\\;-:'), "")
        line = line.translate(translation_table)
        translation_table = dict.fromkeys(map(ord, '=+/#*~&?%!@$.,<>()[]{}0123456789'), "")
        line = line.translate(translation_table)
        translation_table = dict.fromkeys(map(ord, '\''), " \'")
        line = line.translate(translation_table)

        sentence = Sentence(line, 1)

        words = []
        s = line.split(' ')
        for word in s:
            if word is '':
                pass
            else:
                words.append(word)
                word_list.append(word)

        if len(words) < 5:
            continue

        sentence.set_words(words)
        sentence_list.append(sentence)

    for line in neg_list:
        translation_table = dict.fromkeys(map(ord, '\\;-:'), "")
        line = line.translate(translation_table)
        translation_table = dict.fromkeys(map(ord, '=+/#*~&?%!@$.,<>()[]{}0123456789'), "")
        line = line.translate(translation_table)
        translation_table = dict.fromkeys(map(ord, '\''), " \'")
        line = line.translate(translation_table)

        sentence = Sentence(line, 0)

        words = []
        s = line.split(' ')
        for word in s:
            if word is '':
                pass
            else:
                words.append(word)
                word_list.append(word)
        
        if len(words) < 5:
            continue

        sentence.set_words(words)
        sentence_list.append(sentence)

        if len(sentence_list) == 10000:
            break
            
    print("size of sentence : ")
    print(len(sentence_list))

    word_list = list(set(word_list))

    print("size of word : ")
    print(len(word_list))

    i = 0
    for word in word_list:
        w2i[word] = i
        i += 1

    return sentence_list, w2i

class Filter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.filter_tensor = torch.randn(window_size, DIMENTION) / (DIMENTION**0.5)
        self.convol = None
        self.embedding = None
        self.selected = None

    def ReLU(self, x):
        if x < 0:
            return 0.0
        else:
            return x

    def convolution(self, embedding):
        self.embedding = embedding
        embedding_size = embedding.shape[0]

        convol = torch.zeros(embedding_size - self.window_size + 1, 1)
        for i in range(embedding_size - self.window_size + 1):
            embedding_clip = embedding[i:self.window_size+i]
            result = torch.sum(torch.mul(embedding_clip, self.filter_tensor)) / (self.window_size * DIMENTION)
            result = torch.as_tensor(self.ReLU(result))
            convol[i] = result
        self.convol = convol
        return convol
        
    def max_pool(self):
        max = torch.max(self.convol)
        selected = 0
        for value in self.convol:
            if max == value:
                self.selected = selected
            selected += 1

    def backprop(self, learning_rate, grad, mode):
        grad = self.ReLU(grad)
        nomal = grad / (self.window_size * DIMENTION)
        grad_filter = self.embedding[self.selected:self.window_size+self.selected] * nomal

class NN:
    def __init__(self, filter_len, class_num):
        self.output = None
        self.weight = torch.randn(filter_len, class_num) / (class_num**0.5)
        self.loss = None
    
    def forward(self, input, label):
        pass

def train(sentence_list, w2i):
    W_in = torch.randn(len(w2i), DIMENTION) / (DIMENTION**0.5)
    _filter = []

    for _ in range(NUM_FILTER_FEATURE_MAPS):
        _filter.append(Filter(3))
        _filter.append(Filter(4))
        _filter.append(Filter(5))

    for _ in range(EPOCH):
        sentences = sentence_list
        for _ in range(int(len(sentence_list) / TRAIN_BATCH_SIZE)):
            train_sentences = random.sample(sentences, TRAIN_BATCH_SIZE)
            out2 = [s.words for s in train_sentences]
            train_set = []
            for word_list in out2:
                words = []
                for word in word_list:
                    words.append(w2i[word])
                train_set.append(words)
            target_set = [s.state for s in train_sentences]

            for data in train_set:
                for i in range(len(_filter)):
                    convol = _filter[i].convolution(W_in[data])
                    _filter[i].max_pool()

            sentences = [x for x in sentences if x not in train_sentences]



def main():
    sentence_list, w2i = preprocessing()
    
    train(sentence_list, w2i)
    

    #m == torch.nn.Convid(16, 33, 2, stride=2)
    



main()