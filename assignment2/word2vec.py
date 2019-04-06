import torch
from random import shuffle
from collections import Counter
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
################################  Input  ################################
# centerWord : Index of a centerword (type:int)                         #
# contextWord : Index of a contextword (type:int)                       #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))         #
# outputMatrix : Weight matrix of output (type:torch.tesnor(V,D))       #
#########################################################################
    V = inputMatrix.shape[0]
    D = inputMatrix.shape[1]

    h = inputMatrix[centerWord].reshape(D, 1)
    #h.shape = (D,1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (V, 1)
    
    loss = 0
    for i in range(V):
        if i == contextWord:
            loss -= torch.log(softmax[i] + 0.001)
            e[i] = e[i] - 1
        else:
            temp = 1 - softmax[i]
            loss -= torch.log(temp + 0.001)
    
    grad_in = torch.mm(e.reshape(1, V), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(e, h.reshape(1, D))
    #grad_out.shape = (V, D)
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_in, grad_out

def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):
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
    h = h.reshape(D, 1)
    #h.shape = (D,1)
    
    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (V, 1)
    
    loss = 0
    
    for i in range(V):
        if i == centerWord:
            loss -= torch.log(softmax[i] + 0.001)
            e[i] = e[i] - 1
        else:
            temp = 1 - softmax[i]
            loss -= torch.log(temp + 0.001)
            
    grad_in = torch.mm(e.reshape(1, V), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(e, h.reshape(1, D))
    #grad_out.shape = (V, D)
###############################  Output  ################################
# loss : Loss value (type:torch.tensor(1))                              #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))            #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(V,D))          #
#########################################################################

    return loss, grad_in, grad_out


def word2vec_trainer(train_seq, numwords, stats, mode="CBOW", dimension=100, learning_rate=0.0025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]

    print("# of training samples")
    if mode=="CBOW":
    	print(len(train_seq))
    elif mode=="SG":
    	print(len(train_seq)*len(train_seq[0][1]))
    print()

    for _ in range(epoch):
        #Random shuffle of training data
        shuffle(train_seq)
        #Training word2vec using SGD(Batch size : 1)
        for center, contexts in train_seq:
            i+=1
            centerInd = center
            contextInds = contexts
            if mode=="CBOW":
                L, G_in, G_out = CBOW(centerInd, contextInds, W_in, W_out)
                
                W_in[contextInds] -= learning_rate*G_in
                W_out -= learning_rate*G_out

                losses.append(L.item())
            elif mode=="SG":
            	for contextInd in contextInds:
	                L, G_in, G_out = skipgram(centerInd, contextInd, W_in, W_out)
	                W_in[centerInd] -= learning_rate*G_in.squeeze()
	                W_out -= learning_rate*G_out

	                losses.append(L.item())
            else:
                print("Unkwnown mode : "+mode)
                exit()

            if i%100==0:
            	avg_loss=sum(losses)/len(losses)
            	print("%f : %d / %d Loss : %f" %(i/len(train_seq), i, len(train_seq), avg_loss,))
            	losses=[]

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
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
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

    freqtable = []
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
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    print("build training set...")
    #Make tuples of (centerword, contextwords) for training
    train_set = []
    window_size = 5
    for j in range(len(words)):
        if j<window_size:
            contextlist = [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
            train_set.append((w2i[words[j]],contextlist))
        elif j>=len(words)-window_size:
            contextlist = [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
            train_set.append((w2i[words[j]],contextlist))
        else:
            contextlist = [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]
            train_set.append((w2i[words[j]],[w2i[words[j-1]],w2i[words[j+1]]]))

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    emb,_ = word2vec_trainer(train_set, len(w2i), freqtable, mode=mode, dimension=64, epoch=1, learning_rate=0.05)
    
    #Print similar words
    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    for tw in testwords:
    	sim(tw,w2i,i2w,emb)

main()
