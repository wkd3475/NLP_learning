import csv
import re

def remove_words(target):
    translation_table = dict.fromkeys(map(ord, '\\'), " ")
    target = target.translate(translation_table)
    translation_table = dict.fromkeys(map(ord, '-!@$'), "")
    target = target.translate(translation_table)
    return target

def preprocessing(target):

    f = open(target, mode = 'r', newline='')

    lists = csv.reader(f)
    
    output = open(target[:len(target)-4]+'_lower.csv', mode = 'w', encoding='utf-8')
    wr = csv.writer(output)

    for list_ in lists:
        title = remove_words(list_[1].lower())
        #print(title)
        context = remove_words(list_[2].lower())
        #print(context)
        wr.writerow([list_[0], title, context])

    f.close()

preprocessing('test.csv')
preprocessing('train.csv')