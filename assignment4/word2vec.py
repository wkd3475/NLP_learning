import torch
from random import shuffle
from collections import Counter
import argparse
import random
import math
import timeit
import time
import operator

def Analogical_Reasoning_Task(embedding, w2i, i2w, ngram2i, i2ngram, vocab, word_vector_dic, vocab_tokenized):
#######################  Input  #########################
# embedding : Word embedding (type:torch.tesnor(N,D))   #
#########################################################
    testwords = ["narrow-mindedness", "department", "campfires", "knowing", "urbanize", "imperfection", "principality", "abnormal", "secondary", "ungraceful"]
    _, test_tokenized_dic = n_gram_tokenize(testwords)
    #word_tokenized = ['word' : list of ngrams]
    for word, ngrams in test_tokenized_dic.items():
        ngram_list = []
        for ngram in ngrams:
            if ngram in vocab_tokenized:
                ngram_list.append(ngram2i[ngram])
            else:
                pass
        #print(ngram_list)
        #vec : (1:D)
        vec = embedding[ngram_list].sum(0)
        #print(vec)
        print("=========================")
        print("x : %s" %(word))
        Cosine_Similarity_test(vec, word_vector_dic, w2i, vocab)
    

def Cosine_Similarity_test(x, word_vector_dic, w2i, vocab):
    sim_word = {}
    for word in vocab:
            target = word_vector_dic[w2i[word]]
            #print(target)
            x_norm = (x*x).sum(0)**0.5
            target_norm = (target*target).sum(0)**0.5
        
            sim_ = torch.dot(x, target) / (x_norm * target_norm)

            sim_word[word] = sim_
    #print("x = %s (%f) real : %s, %f" %(most_sim, max, q, t))
    most_sim_word = sorted(sim_word.items(), key=operator.itemgetter(1), reverse=True)
    print(most_sim_word[:5])
    print("=========================")
    return most_sim_word[:5]


def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(N,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    N = inputMatrix.shape[0]
    D = inputMatrix.shape[1]
    K = outputMatrix.shape[0]

    h = torch.zeros(1, D)
    #print(inputMatrix)
    for i in range(N):
        h += inputMatrix[i]
    h = h.reshape(D, 1)
    #print(h)
    #h.shape = (D,1)
    
    o = torch.mm(outputMatrix, h)
    sigmoid = torch.sigmoid(o)
    #e = torch.exp(-o)
    #sigmoid = 1 / (1 + e)
    #sigmoid.shape = (K, 1)
    loss = 0
    pos_sum = 0
    neg_sum = -1
    for i in range(K):
        if i is 0:
            pos_sum = -1 * torch.log(sigmoid[i])
            sigmoid[i] -= 1
        else:
            neg_sum = neg_sum * torch.log(1-sigmoid[i])
    loss = pos_sum + neg_sum
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

def word2vec_trainer(input_seq, target_seq, numwords, stats, ngram2i, i2ngram, w_ngram_dic, NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(len(ngram2i), dimension) / (dimension**0.5)
    #(N,D)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    #(V,D)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()

    start = timeit.default_timer()
    #print(dic_activated)
    if NS == 0:
        print("error NS = 0")
    else:
        for _ in range(epoch):
            #Training word2vec using SGD(Batch size : 1)
            for inputs, output in zip(input_seq,target_seq):
                i+=1
                #print(inputs)
                #print(output)
                activated = [output]
                rand = random.choices(stats, k=NS)
                #print(rand)
                for k in range(1, NS+1):
                    if rand[k-1] in activated:
                        pass
                    else:
                        activated.append(rand[k-1])
                L, G_in, G_out = skipgram_NS(inputs, W_in[w_ngram_dic[inputs]], W_out[activated])
                W_in[w_ngram_dic[inputs]] -= learning_rate*G_in.squeeze()
                W_out[activated] -= learning_rate*G_out
                
                losses.append(L.item())
                if i%50000==0:
                    avg_loss=sum(losses)/len(losses)
                    print("i : %d Loss : %f" %(i, avg_loss,))
                    losses=[]

    stop = timeit.default_timer()
    print("Computing time =>")
    print(stop - start)
    return W_in, W_out

def fnv32a( str ):
    hval = 0x811c9dc5
    fnv_32_prime = 0x01000193
    uint32_max = 2 ** 32
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % uint32_max
    return hval

def n_gram_tokenize(vocabs):
    n_gram_vocab = []
    n_gram_dic = {}
    for word_ in vocabs:
        word = "<" + word_ + ">"
        n_gram_dic[word_] = []

        l = len(word)
        #2~6_gram
        for i in range(1, 6):
            if l - i > 0:
                for j in range(l - i):
                    if word[j:j+i+1] not in n_gram_vocab:
                        n_gram_vocab.append(word[j:j+i+1])
                        #n_gram_vocab.append(fnv32a(word[j:j+i+1]))
                    n_gram_dic[word_].append((word[j:j+i+1]))
                    #n_gram_dic[word_].append((fnv32a(word[j:j+i+1])))
        #special
        if l - 6 > 0:
            if word not in n_gram_vocab:
                n_gram_vocab.append(word)
                #n_gram_vocab.append(fnv32a(word))
            n_gram_dic[word_].append(word)
            #n_gram_dic[word_].append(fnv32a(word))
    return n_gram_vocab, n_gram_dic

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
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:10000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()

    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

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
    
    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    #N-gram tokens
    print("N-gram tokenizing...")
    vocab_tokenized, vocab_tokenized_dic = n_gram_tokenize(list(vocab))
    #print(vocab_tokenized_dic)

    #subemb,_ = word2vec_trainer(input_set, target_set, len(w2i), freqtable, NS=ns, dimension=64, epoch=1, learning_rate=0.01)
    ngram2i = {}
    ngram2i[" "] = 0
    i = 1
    for ngram in vocab_tokenized:
        ngram2i[ngram] = i
        i+=1
    i2ngram = {}
    for k,v in ngram2i.items():
        i2ngram[v] = k
    #print(vocab_tokenized_dic)
    w_ngram_dic = {}
    for k, v in vocab_tokenized_dic.items():
        print(k)
        print(v)
        w_ngram_dic[w2i[k]] = []
        for n in v:
            w_ngram_dic[w2i[k]].append(ngram2i[n])

    #print(w_ngram_dic)

    #Make training set
    print("build training set...")
    target_set = []
    input_set = []
    window_size = 5

    #Skipgram
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
    emb,emb2 = word2vec_trainer(input_set, target_set, len(w2i), freqtable, ngram2i, i2ngram, w_ngram_dic, NS=ns, dimension=64, epoch=1, learning_rate=0.01)
    #emb : (N, D)
    torch.save(emb, 'W_in.pth')
    torch.save(emb2, 'W_out.pth')
    word_vector_dic = {}
    for k, v in w_ngram_dic.items():
        word_vector_dic[k] = emb[v].sum(0)

    Analogical_Reasoning_Task(emb, w2i, i2w, ngram2i, i2ngram, vocab, word_vector_dic, vocab_tokenized)

main()
