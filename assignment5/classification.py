import csv
import pickle
import torch
import timeit

def preprocessing(target):
    translation_table = dict.fromkeys(map(ord, '\\;'), " ")
    target = target.translate(translation_table)
    translation_table = dict.fromkeys(map(ord, '-!@$.,'), "")
    target = target.translate(translation_table)
    return target

def bigram_tokenize(data_set):
    news = []
    #i2class : sentence num -> class
    i2class = {}

    s2i = {}
    i = 0
    print("- step1 : making s2i...")
    for list_ in data_set:
        list_[2] = preprocessing(list_[2].lower())
        news.append(list_[2])
        #sentence num -> class
        i2class[i] = int(list_[0]) - 1
        #sentence -> num
        s2i[list_[2]] = i
        i += 1

    #bigram list
    bigram = []

    bigram2i = {}
    i2bigram = {}
    s_bag_temp = {}
    s_bag = {}

    print("- step2 : making bigram...")
    for sentence in news:
        corpus = sentence.split()
        s_bag_temp[s2i[sentence]] = []
        for i in range(len(corpus)-1):
            temp = corpus[i] + '-' + corpus[i+1]
            s_bag_temp[s2i[sentence]].append(temp)
            if temp in bigram:
                pass
            else:
                bigram.append(temp)
    
    print("- step3 : making bigram2i, i2bigram, s_bag...")
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

    return s2i, i2class, bigram2i, i2bigram, s_bag, bigram

#target = class number
#inputs = bag of bigram numbers
#inputMatrix = (N,D)
#outputMatrix = (4,D)
def classification(target, inputs, inputMatrix, outputMatrix):
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

    loss = -torch.log(softmax[target])
    softmax[target] = softmax[target] - 1

    grad_in = torch.mm(softmax.reshape(1, 4), outputMatrix)
    #grad_in.shape = (1, D)
    grad_out = torch.mm(softmax, h.reshape(1, D))
    #grad_out.shape = (4, D)

    return loss, grad_in, grad_out

def trainer(input_set, bigram2i, s_bag, dimension=64, learning_rate=0.01, epoch=1):
    W_in = torch.randn(len(bigram2i), dimension) / (dimension**0.5)
    #(N,D) N : nuber of bigrams, D : dimension
    W_out = torch.randn(4, dimension) / (dimension**0.5)
    #(4,D)

    i = 0
    losses = []
    print("# of training samples")
    print(len(input_set))
    print()

    for _ in range(epoch):
        #target : class number, input_sentence : int
        for target, input_sentence in input_set:
            i += 1
            inputs = s_bag[input_sentence]
            L, G_in, G_out = classification(target, inputs, W_in, W_out)
            W_in[inputs] -= learning_rate*G_in.squeeze()
            W_out -= learning_rate*G_out

            losses.append(L.item())

            if i%10000 == 0:
                avg_loss=sum(losses)/len(losses)
                print("%d Loss : %f" %(i, avg_loss,))
                losses = []
    
    return W_in, W_out


def main():
    TRAIN = 50000
    TEST = 5000
    start = timeit.default_timer()
    train_dic = open('train.csv', mode='r', encoding='utf-8').readlines() [:TRAIN]
    test_dic = open('test.csv', mode='r', encoding='utf-8').readlines() [:TEST]
    train_lists = csv.reader(train_dic)
    test_lists = csv.reader(test_dic)

    print("train data : bigram_tokenizing...")
    s2i, i2class, bigram2i, i2bigram, s_bag, bigram = bigram_tokenize(train_lists)

    input_set = []
    for k in s_bag.keys():
        input_set.append([i2class[k], k])

    #############################################################
    print("test data : bigram_tokenizing...")
    s2i_test, i2class_test, bigram2i_test, i2bigram_test, s_bag_test, _ = bigram_tokenize(test_lists)
    test_set = []
    for k in s_bag_test.keys():
        test_set.append([i2class_test[k], k])

    print("training...")
    #emb1.shape = (N, D), emb2.shape = (4, D)
    emb1, emb2 = trainer(input_set, bigram2i, s_bag, dimension=64, learning_rate=0.01, epoch=1)

    print("testing...")
    print("# of tesing samples")
    print(len(s2i_test))
    print()
    total = 0
    correct = 0

    j = 0
    for target, test_sentence in test_set:
        j += 1
        h = torch.zeros(1, 64)
        inputs = s_bag_test[test_sentence]
        for i in inputs:
            b = i2bigram_test[i]
            if b in bigram:
                h += emb1[bigram2i[b]]
        
        o = torch.mm(h, emb2.reshape(64, 4))
        e = torch.exp(o - torch.max(o))
        softmax = e / torch.sum(e)
        result = torch.argmax(softmax)
        target = target - 1
        t = torch.tensor(target)

        if torch.equal(t, result):
            correct += 1
        total += 1
        
        if j%1000 == 0:
            print("%d / %d ..." %(j, len(s2i_test)))
    stop = timeit.default_timer()
    print("==============================================")
    print("train_data : %d" %TRAIN)
    print("test_data : %d" %TEST)
    print("computing time : %.2f" %(stop-start))
    print("correct / total = %d / %d" %(correct, total))
    print("accuracy = %.2f" %(correct/total * 100))
    print("==============================================")



main()