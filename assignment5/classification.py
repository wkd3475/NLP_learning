import csv
import pickle
import torch
import timeit

DIMEN = 256
EPOCH = 3
RATE = 0.03

def preprocessing(target):
    translation_table = dict.fromkeys(map(ord, '\\;-:'), "")
    target = target.translate(translation_table)
    translation_table = dict.fromkeys(map(ord, '=+/#*~&?%!@$.,<>()[]{}0123456789'), "")
    target = target.translate(translation_table)
    translation_table = dict.fromkeys(map(ord, '\''), " \'")
    target = target.translate(translation_table)

    return target

def test_bigram_tokenize(data_set, bigram2i):
    result = []
    for list_ in data_set:
        list_[2] = preprocessing(list_[2].lower())
        corpus = list_[2].split(' ')

        bigram = []
        for j in range(len(corpus)-1):
            temp = corpus[j] + '-' + corpus[j+1]
            if temp in bigram2i:
                bigram.append(bigram2i[temp])
        bigram = list(set(bigram))
        result.append([int(list_[0])-1, bigram, list_[2]])
    
    return result

def bigram_tokenize(data_set):
    news = []
    #i2class : sentence num -> class
    i2class = {}

    s2i = {}
    i2s = {}
    i = 0
    #print("- step1 : making s2i, i2s...")
    for list_ in data_set:
        list_[2] = preprocessing(list_[2].lower())
        if list_[2] in s2i:
            pass
        else:
            news.append(list_[2])
            i2class[i] = int(list_[0]) - 1
            s2i[list_[2]] = i
            i2s[i] = list_[2]
            i += 1
        
    #bigram list
    bigram = []

    bigram2i = {}
    i2bigram = {}
    s_bag_temp = {}
    s_bag = {}

    #print("- step2 : making bigram, sentence(%d) ..." %(len(news)))
    for sentence in news:
        corpus = sentence.split(' ')
        s_bag_temp[s2i[sentence]] = []
        for j in range(len(corpus)-1):
            temp = corpus[j] + '-' + corpus[j+1]
            s_bag_temp[s2i[sentence]].append(temp)
            bigram.append(temp)
    
    bigram = list(set(bigram))

    #print("- step3 : making bigram2i, i2bigram, s_bag...")
    i = 0
    for v in bigram:
        bigram2i[v] = i
        i2bigram[i] = v
        i += 1

    for k,v in s_bag_temp.items():
        s_bag[k] = []
        #b = bigram
        for b in v:
            s_bag[k].append(bigram2i[b])
    return s2i, i2class, bigram2i, s_bag

#target = class number
#inputs = bag of bigram numbers
#inputMatrix = (N,D)
#outputMatrix = (4,D)
def classification(target, inputs, inputMatrix, outputMatrix):
    list_ = []
    N = inputMatrix.shape[0]
    D = inputMatrix.shape[1]

    h = torch.zeros(1, D)

    for bigram in inputs:
        h += inputMatrix[bigram]
    h = h.reshape(D, 1)

    o = torch.mm(outputMatrix, h)
    e = torch.exp(o - torch.max(o))
    softmax = e / torch.sum(e)
    #softmax.shape = (4, 1)
    result = torch.argmax(softmax)
    t = torch.tensor(target)
    predict = None
    if torch.equal(result, t):
        predict = 1
    else:
        predict = 0
    
    loss = -torch.log(softmax[target])
    softmax[target] = softmax[target] - 1

    grad_in = torch.mm(softmax.reshape(1, 4), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(softmax, h.reshape(1, D))
    #grad_out.shape = (4, D)
    
    list_ = [target, int(t)]
    return loss, grad_in, grad_out, predict, list_

def trainer(input_set, bigram2i, s_bag, dimension=64, learning_rate=0.01, epoch=1):
    W_in = torch.randn(len(bigram2i), dimension) / (dimension**0.5)
    #(N,D) N : nuber of bigrams, D : dimension
    W_out = torch.randn(4, dimension) / (dimension**0.5)
    #(4,D)

    i = 0
    losses = []
    acc = []
    print("# of training samples")
    print(len(input_set))
    print()

    for _ in range(epoch):
        #target : class number, input_sentence : int
        for target, input_sentence in input_set:
            i += 1
            inputs = s_bag[input_sentence]
            L, G_in, G_out, predict, _= classification(target, inputs, W_in, W_out)
            W_in[inputs] -= learning_rate*G_in
            W_out -= learning_rate*G_out

            losses.append(L.item())

            if predict is 1:
                acc.append(1)
            else:
                acc.append(0)

            if i%(50000*epoch) == 0:
                avg_loss=sum(losses)/len(losses)
                print("(%d / %d) Loss : %f" %(i, len(input_set) * epoch, avg_loss,))
                losses = []
                print("train_set accuracy : %.2f" %(sum(acc)/len(acc) * 100))
                acc = []

    avg_loss=sum(losses)/len(losses)
    print("(%d / %d) Loss : %f" %(i, len(input_set) * epoch, avg_loss,))
    print("train_set accuracy : %.2f" %(sum(acc)/len(acc) * 100))
    print()
    return W_in, W_out


def main():
    start = timeit.default_timer()
    train_dic = open('train.csv', mode='r', encoding='utf-8').readlines()
    test_dic = open('test.csv', mode='r', encoding='utf-8').readlines()

    train_lists = csv.reader(train_dic)
    test_lists = csv.reader(test_dic)

    print("train data : bigram_tokenizing...")
    s2i, i2class, bigram2i, s_bag = bigram_tokenize(train_lists)

    input_set = []
    for k in s_bag.keys():
        input_set.append([i2class[k], k])

    print("test data : bigram_tokenizing...")
    test_set = test_bigram_tokenize(test_lists, bigram2i)

    print()
    print("training...")
    #emb1.shape = (N, D), emb2.shape = (4, D)
    emb1, emb2 = trainer(input_set, bigram2i, s_bag, dimension=DIMEN, learning_rate=RATE, epoch=EPOCH)

    print("testing...")
    print("# of tesing samples")
    print(len(test_set))
    print()
    
    acc = []
    f = open("result.txt", "w")
    i = 0 
    for test in test_set:
        i += 1
        _, _, _, predict, list_ = classification(test[0], test[1], emb1, emb2)
        f.write("sentence %d : %s\npredict : %d, real : %d\n" %(i, test[2], list_[0], list_[1]))
        if predict is 1:
            acc.append(1)
        else:
            acc.append(0)
    
    stop = timeit.default_timer()
    print("==============================================")
    print("train_data : all")
    print("test_data : all")
    print("# of bigram : %d" %(len(bigram2i)))
    print("epoch : %d" %EPOCH)
    print("dimen : %d" %DIMEN)
    print("learning_rate : %.3f" %RATE)
    print("computing time : %.2f" %(stop-start))
    print("correct / total : %d / %d" %(sum(acc), len(acc)))
    print("test_set accuracy : %.2f" %(sum(acc)/len(acc) * 100))
    print("==============================================")
            
main()