import torch
from random import shuffle
from collections import Counter
import argparse
from huffman import HuffmanCoding
import random
import math
import timeit
import time
import operator

def Analogical_Reasoning_Task(embedding, w2i, i2w, qlist, vocab):
#######################  Input  #########################
# embedding : Word embedding (type:torch.tesnor(V,D))   #
#########################################################
    simdict = {}
    num = 0
    num7 = 0
    numC = 0
    sum = 0
    for q in qlist[:100]:
        w1 = w2i[q[0]]
        v1 = embedding[w1]
        w2 = w2i[q[1]]
        v2 = embedding[w2]
        w3 = w2i[q[2]]
        v3 = embedding[w3]
        w4 = w2i[q[3]]
        v4 = embedding[w4]

        x = v2 + v3 - v1

        # x = q[3] ?
        if Cosine_Similarity_test(x, q, embedding, w2i, vocab):
            numC += 1

        v4_norm = (v4*v4).sum(0)**0.5
        x_norm = (x*x).sum(0)**0.5
    
        sim_ = torch.dot(v4, x) / (v4_norm * x_norm)
        sum += sim_
        if sim_ > 0.3:
            num += 1
        
        if sim_ > 0.7:
            num7 += 1

    print("total number : %d" %len(qlist))
    print("number of sim(>0.3) : %d" %num)
    print("number of sim(>0.7) : %d" %num7)
    print("value : %f%%" %(sum/len(qlist) * 100))
    print("correct x (in top 100): %d / %d = %f%%" %(numC, len(qlist), numC/len(qlist)*100))
        
def Cosine_Similarity_test(x, q, embedding, w2i, vocab):
    target = q[3]
    D = embedding.shape[1]
    sim_word = {}
    for word in vocab:
        if word == q[0] or word == q[1] or word == q[2]:
            pass
        else:
            v = embedding[w2i[word]]
            v_norm = (v*v).sum(0)**0.5
            x_norm = (x*x).sum(0)**0.5
        
            sim_ = torch.dot(v, x) / (v_norm * x_norm)

            sim_word[word] = sim_
    #print("x = %s (%f) real : %s, %f" %(most_sim, max, q, t))
    most_sim_word = sorted(sim_word.items(), key=operator.itemgetter(1), reverse=True)
    #print(most_sim_word[0:10])
    for i in range(100):
        #print(most_sim_word[i][0])
        if target == most_sim_word[i][0]:
            return True
    else:
        return False

def subsampling(word_seq):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################
    stats = Counter(word_seq)
    N = len(word_seq)

    f = stats
    for word in stats:
        f[word] = f[word] / N
    
    subsampled = []
    for word in word_seq:
        Pd = 1 - (0.00001 / f[word])**0.5
        if random.uniform(0, 1) > Pd:
            subsampled.append(word)

    return subsampled

def skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
###################################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]

    h = inputMatrix[centerWord].reshape(D, 1)
    #h.shape = (D,1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (V, 1)

    loss = -torch.log(softmax[contextWord]+0.00001)
    softmax[contextWord] = softmax[contextWord] - 1


    grad_in = torch.mm(softmax.reshape(1, V), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(softmax, h.reshape(1, D))
    #grad_out.shape = (V, D)

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))                    #
###################################################################################

    return loss, grad_in, grad_out

def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]

    h = inputMatrix[centerWord].reshape(D, 1)
    #h.shape = (D,1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(-o)
    sigmoid = 1 / (1 + e)
    #sigmoid.shape = (K, 1)

    p = 1
    code = list(contextCode)
    #print(outputMatrix)
    #print("code : %s" %code)
    for i in range(K):
        if code[i] == '0':
            p *= sigmoid[i]
            sigmoid[i] = sigmoid[i] - 1

        elif code[i] == '1':
            p *= (1 - sigmoid[i])
    #time.sleep(3)
    loss = -torch.log(p)

    grad_in = torch.mm(sigmoid.reshape(1, K), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(sigmoid, h.reshape(1, D))
    #grad_out.shape = (K, D)

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    return loss, grad_in, grad_out



def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]
    h = inputMatrix[centerWord].view(D,1)
    #h.shape = (D,1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(-o)
    sigmoid = 1 / (1 + e)
    #sigmoid.shape = (K, 1)

    loss = 0
    neg_sum = 1
    for i in range(K):
        if i is 0:
            loss -= torch.log(sigmoid[i])
            sigmoid[i] -= 1
        else:
            neg_sum *= torch.log(1-sigmoid[i])
    loss = loss - neg_sum

    grad_in = torch.mm(sigmoid.view(1,-1), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(sigmoid, h.reshape(1, -1))
    #grad_out.shape = (K, D)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    return loss, grad_in, grad_out

def CBOW(contextWords, centerWord, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWords : Indices of contextwords (type:list(int))               #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]

    h = torch.zeros(1, D)
    for word in contextWords:
        h = h + inputMatrix[word]
    h = h.view(D,1)
    #h.shape = (D,1)
    
    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (V, 1)
    
    loss = -torch.log(softmax[centerWord]+0.00001)
    softmax[centerWord] = softmax[centerWord] - 1
    
    grad_in = torch.mm(softmax.view(1,-1), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(softmax, h.view(1,-1))
    #grad_out.shape = (V, D)
    
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################
    return loss, grad_in, grad_out

def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# centerCode : Code of a centerword (type:str)                                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]

    h = torch.zeros(1, D)
    for word in contextWords:
        h = h + inputMatrix[word]
    h = h.reshape(D, 1)
 
    o = torch.mm(outputMatrix, h)
    e = torch.exp(-o)
    sigmoid = 1 / (1 + e)
    #sigmoid.shape = (K, 1)

    p = 1
    code = list(centerCode)
    for i in range(K):
        if code[i] == '0':
            p *= sigmoid[i]
            sigmoid[i] = sigmoid[i] - 1
        elif code[i] == '1':
            p *= 1 - sigmoid[i]

    loss = -torch.log(p)

    grad_in = torch.mm(sigmoid.reshape(1, K), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(sigmoid, h.reshape(1, D))
    #grad_out.shape = (K, D)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]
    
    h = torch.zeros(1, D)
    for word in contextWords:
        h = h + inputMatrix[word]
    h = h.reshape(D, 1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(-o)
    sigmoid = 1 / (1 + e)
    #sigmoid.shape = (K, 1)

    loss = 0
    neg_sum = 1
    for i in range(K):
        if i is 0:
            loss == loss - torch.log(sigmoid[i])
            sigmoid[i] -= 1
        else:
            neg_sum *= torch.log(1-sigmoid[i])
    loss = loss - neg_sum

    grad_in = torch.mm(sigmoid.reshape(1, K), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(sigmoid, h.reshape(1, D))
    #grad_out.shape = (K, D)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    return loss, grad_in, grad_out

############################# Huffman_code => Tree ################################
class Node():
    def __init__(self, num):
        self.num = num
        self.left = None
        self.right = None

class Leaf():
    def __init__(self, value):
        self.value = value

class Tree():
    def __init__(self,):
        self.root = Node(0)
        self.num = 1
        self.activated = {}

    def get_activated(self, codes):
        for val, code in codes.items():
            parent = self.root
            l = len(code)
            code_ = list(code)
            self.activated[val] = []
            for i in range(l):
                self.activated[val].append(parent.num)
                if i < l-1:
                    if code_[i] == '0' and parent.left is None:
                        parent.left = Node(self.num)
                        parent = parent.left
                        self.num += 1
                    elif code_[i] == '0' and parent.left is not None:
                        parent = parent.left
                    elif code_[i] == '1' and parent.right is None:
                        parent.right = Node(self.num)
                        parent = parent.right
                        self.num += 1
                    elif code_[i] == '1' and parent.right is not None:
                        parent = parent.right
                elif i == l-1:
                    if code_[i] == '0':
                        parent.left = Leaf(val)
                    elif code_[i] == '1':
                        parent.right = Leaf(val)
        #print(self.num)
        return self.activated
###################################################################################


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()

    if NS is 0:
        tree = Tree()
        dic_activated = tree.get_activated(codes)

    start = timeit.default_timer()
    #print(dic_activated)
    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1

            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    activated = dic_activated[output]
                    print("output : %s, activated : %s" %(output, activated))
                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                elif NS==-1:
                    L, G_in, G_out = CBOW(inputs, output, W_in, W_out)
                    W_in[inputs] -= learning_rate*G_in
                    W_out -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    activated = [output]
                    rand = random.choices(stats, k=NS)
                    for k in range(1, NS+1):
                        if rand[k] in activated:
                            pass
                        else:
                            activated.append(rand[k])

                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    activated = dic_activated[output]
                    #print("inputs : %s" %inputs)
                    #print("output : %s, activated : %s" %(output, activated))
                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                elif NS==-1:
                    L, G_in, G_out = skipgram(inputs, output, W_in, W_out)
                    W_in[inputs] -= learning_rate*G_in
                    W_out -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    activated = [output]
                    rand = random.choices(stats, k=NS)
                    for k in range(1, NS+1):
                        if rand[k-1] in activated:
                            pass
                        else:
                            activated.append(rand[k-1])
                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L.item())
            if i%50000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("i : %d Loss : %f" %(i, avg_loss,))
            	losses=[]
    stop = timeit.default_timer()
    print("Computing time =>")
    print(stop - start)
    return W_in, W_out


def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()

def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:50000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()

    ##Subsampling
    #subsampling(corpus)

    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Quesions-words
    qlist = []
    question = []
    questions = open('questions-words.txt',mode='r').read().splitlines()
    for i in range(len(questions)):
        question.append(questions[i].split())
    for q in question:
        if len(q) is not 4:
            pass
        else:
            check = True
            if q[0] not in vocab:
                check = False
            if q[1] not in vocab:
                check = False
            if q[2] not in vocab:
                check = False
            if q[3] not in vocab:
                check = False
            if check:
                qlist.append(q)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    #Code dict for hierarchical softmax
    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
    
    codedict = HuffmanCoding().build(freqdict)
    #print(codedict)
    
    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    #Make training set
    print("build training set...")
    target_set = []
    input_set = []
    window_size = 5
    if mode=="CBOW":
        for j in range(len(words)):
            if j<window_size:
                input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
            elif j>=len(words)-window_size:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                target_set.append(w2i[words[j]])
            else:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
    if mode=="SG":
        for j in range(len(words)):
            if j<window_size:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
            elif j>=len(words)-window_size:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
            else:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    emb, emb2 = word2vec_trainer(input_set, target_set, len(w2i), codedict, freqtable, mode=mode, NS=ns, dimension=64, epoch=1, learning_rate=0.01)
    
    Analogical_Reasoning_Task(emb, w2i, i2w, qlist, vocab)

    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    for tw in testwords:
    	sim(tw,w2i,i2w,emb)

main()
