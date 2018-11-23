import pandas as pd
import numpy as np
from sklearn import tree
from scipy import spatial
import csv
import numpy as np

base_dir = './Quora_question_pair_partition/'
train_dir = base_dir + 'train.tsv'
dev_dir = base_dir + 'dev.tsv'
test_dir = base_dir + 'test.tsv'
word_dir = base_dir + 'wordvec.txt'
words_vec = {}

def loadWordDic():
    with open(word_dir, 'r') as words:
        tweets_reader = csv.reader(words, delimiter=' ')
        cnt = 0
        for row in tweets_reader:
            words_vec[row[0]] = np.array([float(x) for x in row[1:]]).reshape((1,300))
            cnt += 1
            if cnt%5000 == 0:
                print(cnt)
loadWordDic()

# read data
def loadData():
    train_df = pd.read_csv(train_dir,sep='\t', header=None, names=['label','q1','q2','id'])
    train_X, train_y = train_df[['q1','q2']], train_df['label']
    dev_df = pd.read_csv(dev_dir,sep='\t', header=None, names=['label','q1','q2','id'])
    dev_X, dev_y = dev_df[['q1','q2']], dev_df['label']
    test_df = pd.read_csv(test_dir,sep='\t', header=None, names=['label','q1','q2','id'])
    test_X, test_y = test_df[['q1','q2']], test_df['label']
    return train_X, train_y, dev_X, dev_y, test_X, test_y


def generate_features(df):
    feature_names = []

    # number of words
    df['cnt'] = df.apply(lambda row: abs(len(row['q1'].split(' ')) - len(row['q2'].split(' '))), axis=1)
    feature_names.append('cnt')

    # distance between sentences
    #df['dis'] = df.apply(dis, axis=1)
    #feature_names.append('dis')

    # magic feature: question frequency

    # same word frequency
    df['fre1'] = df.apply(fre1, axis=1)
    feature_names.append('fre1')
    df['fre2'] = df.apply(fre2, axis=1)
    feature_names.append('fre2')

    ''' try to add more than one columns one time but failed
    temp = df.apply(fre3, axis=1)
    temp.columns = ['fre1', 'fre2']
    df.join(temp.to_frame())
    df[['q1','q2']] = df.apply(fre3, axis=1)
    feature_names.append('q1')
    feature_names.append('q2')
    '''

    # cosine based distance statistics
    print("1\n")
    df['dis_mean1'] = df.apply(dis_mean1, axis=1)
    feature_names.append('dis_mean1')
    #df['dis_core_mean1'] = df.apply(dis_core_mean1, axis=1)
    #feature_names.append('dis_core_mean1')
    #df['dis_median1'] = df.apply(dis_median1, axis=1)
    #feature_names.append('dis_median1')
    print("2\n")
    df['dis_mean2'] = df.apply(dis_mean2, axis=1)
    feature_names.append('dis_mean2')
    #df['dis_core_mean2'] = df.apply(dis_core_mean2, axis=1)
    #feature_names.append('dis_core_mean2')
    #df['dis_median2'] = df.apply(dis_median2, axis=1)
    #feature_names.append('dis_median2')

    # relative length rate
    #df['rate_len'] = df.apply(rate_len, axis=1)
    #feature_names.append('rate_len')

    print("Finish generating features")

    return feature_names, df


## train model
def train(train_X, train_y, para):
    feature_names, df = generate_features(train_X)

    clf = tree.DecisionTreeClassifier(max_depth=para['max_depth'])

    clf.fit(df[feature_names], train_y)

    acc = clf.score(df[feature_names], train_y)

    print("Accuracy on the training set:", acc)

    return acc, clf


## select hyper parameters on the dev set
def select_model(train_X, train_y, dev_X, dev_y, paras):
    max_acc = 0
    best_model = None
    for para in paras:
        train_acc, model = train(train_X, train_y, para)
        feature_names, df = generate_features(dev_X)
        dev_acc = model.score(df[feature_names], dev_y)
        if dev_acc > max_acc:
            max_acc = dev_acc
            best_model = model
    return max_acc, model

## evaluate on the test set
def test(test_X, test_y, model):
    feature_names, df = generate_features(test_X)
    acc = model.score(df[feature_names], test_y)
    return acc


def dis(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')
    v1 = np.zeros((1,300))
    cnt1 = 0
    v2 = np.zeros((1,300))
    cnt2 = 0
    for word in words1:
        if word.lower() in words_vec:
            cnt1 += 1
            v1 += words_vec[word.lower()]
        #else:
        #    print("Skip word ",word)
    for word in words2:
        if word.lower() in words_vec:
            cnt2 += 1
            v2 += words_vec[word.lower()]
        #else:
        #    print("Skip word",word)
    return (v1 / cnt1 - v2  / cnt2).sum()

def fre1(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')
    same_count = 0
    for word in words1:
        if word in words2:
            same_count += 1

    return (same_count / len(words1))

def fre2(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')
    same_count = 0
    for word in words1:
        if word in words2:
            same_count += 1

    return (same_count / len(words2))

'''
def fre3(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')
    same_count = 0
    for word in words1:
        if word in words2:
            same_count += 1

    return (same_count / len(words2)),(same_count / len(words1))
'''

def dis_mean1(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')

    max_similarity = []
    for word1 in words1:
        temp = 0
        for word2 in words2:
            if (word1.lower() in words_vec and word2.lower() in words_vec):
                value = 1 - spatial.distance.cosine(words_vec[word1.lower()], words_vec[word2.lower()])
                # print(word1 + ' ' + word2 + ' ' + str(value))
                if (value > temp):
                    temp = value
        max_similarity.append(temp)

    return (np.mean(max_similarity))


def dis_median1(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')

    max_similarity = []
    for word1 in words1:
        temp = 0
        for word2 in words2:
            if (word1.lower() in words_vec and word2.lower() in words_vec):
                value = 1 - spatial.distance.cosine(words_vec[word1.lower()], words_vec[word2.lower()])
                # print(word1 + ' ' + word2 + ' ' + str(value))
                if (value > temp):
                    temp = value
        max_similarity.append(temp)

    return (np.median(max_similarity))

def dis_core_mean1(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')

    max_similarity = []
    for word1 in words1:
        temp = 0
        for word2 in words2:
            if (word1.lower() in words_vec and word2.lower() in words_vec):
                value = 1 - spatial.distance.cosine(words_vec[word1.lower()], words_vec[word2.lower()])
                # print(word1 + ' ' + word2 + ' ' + str(value))
                if (value > temp):
                    temp = value
        max_similarity.append(temp)

    max_similarity.sort(reverse=True)
    return (np.mean(max_similarity[:(len(max_similarity) // 2)]))

def dis_mean2(row):
    words1 = row['q2'].split(' ')
    words2 = row['q1'].split(' ')

    max_similarity = []
    for word1 in words1:
        temp = 0
        for word2 in words2:
            if (word1.lower() in words_vec and word2.lower() in words_vec):
                value = 1 - spatial.distance.cosine(words_vec[word1.lower()], words_vec[word2.lower()])
                # print(word1 + ' ' + word2 + ' ' + str(value))
                if (value > temp):
                    temp = value
        max_similarity.append(temp)

    return (np.mean(max_similarity))


def dis_median2(row):
    words1 = row['q2'].split(' ')
    words2 = row['q1'].split(' ')

    max_similarity = []
    for word1 in words1:
        temp = 0
        for word2 in words2:
            if (word1.lower() in words_vec and word2.lower() in words_vec):
                value = 1 - spatial.distance.cosine(words_vec[word1.lower()], words_vec[word2.lower()])
                # print(word1 + ' ' + word2 + ' ' + str(value))
                if (value > temp):
                    temp = value
        max_similarity.append(temp)

    return (np.median(max_similarity))

def dis_core_mean2(row):
    words1 = row['q2'].split(' ')
    words2 = row['q1'].split(' ')

    max_similarity = []
    for word1 in words1:
        temp = 0
        for word2 in words2:
            if (word1.lower() in words_vec and word2.lower() in words_vec):
                value = 1 - spatial.distance.cosine(words_vec[word1.lower()], words_vec[word2.lower()])
                # print(word1 + ' ' + word2 + ' ' + str(value))
                if (value > temp):
                    temp = value
        max_similarity.append(temp)

    max_similarity.sort(reverse=True)
    return (np.mean(max_similarity[:(len(max_similarity) // 2)]))

def rate_len(row):
    words1 = row['q1'].split(' ')
    words2 = row['q2'].split(' ')
    len1 = len(words1)
    len2 = len(words2)

    rate = len1/len2
    if(rate > 1):
        rate = 1/rate

    return rate


def main():
    # read data
    train_X, train_y, dev_X, dev_y, test_X, test_y = loadData()
    print("Finish loading data")

    paras = [{'max_depth': 100}]

    acc_dev, model = select_model(train_X, train_y, dev_X, dev_y, paras)
    print("Accuracy on the development set is:", acc_dev)

    acc_test = test(test_X, test_y, model)
    print("Accuracy on the test set is:", acc_test)

main()