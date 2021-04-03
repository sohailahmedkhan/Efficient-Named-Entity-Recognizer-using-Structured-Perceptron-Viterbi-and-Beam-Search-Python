from collections import Counter
from itertools import product
from collections import defaultdict
from sklearn.metrics import f1_score
import random
import operator
import sys
import time


def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None): 
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()] 
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()] 
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

#Get the word_label counts in the corpus
def get_current_word_current_label_counts(train_data):
    train_set = []
    counts = {}
    for i in range(len(train_data)):
        train_set.extend(train_data[i])
    counts = Counter(train_set)
    
    return counts


def viterbi(words, w, features):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    counts_list = []
    
    best_label = []
    for word in words:
        counts = {}
        best = {}
        #Getting weights for each label 
        for label in labels:
            phi = phi_1([word], [label], features)
            count_phi = 0
            for key in phi:
                count_phi += w[key] * phi[key]

            if counts_list:
                maxVal = -100
                for prev_label in labels:
                
                    count = counts_list[-1][prev_label]
                    count += count_phi
                    if count > maxVal:
                        counts[label] = count
                        maxVal = count
                        best[label] = prev_label
            else:
                counts[label] = count_phi
        counts_list.append(counts)
        best_label.append(best)      
    last_label = max(counts_list[-1].items(), key=operator.itemgetter(1))[0]
    final_labels = [last_label]
    for i in range(len(words)-1):
        final_labels.insert(0,best_label[-1-i][final_labels[-1-i]])
    return final_labels

def beam(words, w, features):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    counts_list = []
    
    best_label = []
    top_labels = []
    for word in words:
        counts = {}
        best = {}
        #Getting weights for each label 
        for label in labels:
            phi = phi_1([word], [label], features)
            count_phi = 0
            for key in phi:
                count_phi += w[key] * phi[key]
            #if counts list is not empty
            if counts_list:
                maxVal = -100
                for prev_label in top_labels:
                    count = counts_list[-1][prev_label]
                    count += count_phi
                    if count > maxVal:
                        counts[label] = count
                        maxVal = count
                        best[label] = prev_label
            #if counts_list is empty
            else:
                counts[label] = count_phi
        counts_list.append(counts)
        #Using Beam Search with Beam = 5, you can change [:5] below to any number less than or equal to 5 to get
        # Beam search for that Beam size
        top_labels = sorted(counts, key=counts.get, reverse=True)[:5]
        best_label.append(best)
    last_label = max(counts_list[-1].items(), key=operator.itemgetter(1))[0]
    final_labels = [last_label]
    for i in range(len(words)-1):
        final_labels.insert(0,best_label[-1-i][final_labels[-1-i]])
    return final_labels

#Implementation for PHI1
def phi_1(words, labels, cw_cl_counts):
    dictionary = defaultdict(int)
    #Making a dictionary with word, labels and their counts
    for i in range(len(words)):
        if (words[i], labels[i]) in cw_cl_counts:
            dictionary[words[i],labels[i]] += 1
        else:
            dictionary[words[i],labels[i]] = 0
    return dictionary

#Perceptron train of PHI1
def phi1_perceptron_train(train_data, features, maxIter, scheme):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    w = defaultdict(int)
    for iterr in range(maxIter):
        print("Iteration #: ", iterr+1, " for Phi1 Train")
        random.shuffle(train_data)
        for sentence in train_data:
            words = []
            #Generating all possible labels
            sentence_labels = []
            #getting all words in sentence in words list
            for word, label in sentence:
                words.append(word)
                sentence_labels.append(label)

            if scheme == '-v':
                predict_label = viterbi(words,w,features)
            elif scheme == '-b':
                predict_label = beam(words,w,features)

            predict_phi = phi_1(words,predict_label,features)
            correct_phi = phi_1(words, sentence_labels, features)
            #Adjust weights
            if predict_label != sentence_labels:

                for key in correct_phi:
                    w[key] += correct_phi[key]

                for key in predict_phi:
                    w[key] -= predict_phi[key]
    return w

def phi1_perceptron_test(test_data, w, features, scheme):
    labels = ["O", "PER", "LOC", "ORG", "MISC"]
    all_possible_labels = []
    #w = defaultdict(int)
    correct = []
    predicted = []
    for sentence in test_data:
        words = []
        all_possible_labels = list(product(labels,repeat = len(sentence)))
        sentence_labels = []
        for word, label in sentence:
            words.append(word)
            sentence_labels.append(label)
        correct.append(sentence_labels)
        #Choosing the Scheme (Viterbi, Beam)
        if scheme == '-v':
            predict_label = viterbi(words,w,features)
        elif scheme == '-b':
            predict_label = beam(words,w,features)
        predicted.append(predict_label)

    #Flatting the lists with correct and predicted labels
    flat_cor = []
    flat_pre = []
    for sublist in correct:
        for item in sublist:
            flat_cor.append(item)
        
    for sublist in predicted:
        for item in sublist:
            flat_pre.append(item)

    return flat_cor, flat_pre

def main():
    #Getting file paths from the command line arguments
    train_path =  sys.argv[2]
    test_path =  sys.argv[3]
    scheme = sys.argv[1]
    flat_cor = []
    flat_pre = []
    maxIter = 5
    train_data = load_dataset_sents(train_path)
    test_data = load_dataset_sents(test_path)
    random.seed(1)
    
    start = time.time()
    if (scheme == '-v'):
        print("\nUsing..." + "\t Viterbi" + " and Using ", maxIter, " Iterations and Seed = 1\n")
    elif (scheme == '-b'):
        print("\nUsing..." + "\t Beam Search" + " and Using ", maxIter, " Iterations and Seed = 1\n")
    else:
        print("\nWrong arguments... Exiting Program\n")
        exit()

    #getting word, tag counts in the corpus
    cw_cl_counts = {}
    cw_cl_counts = get_current_word_current_label_counts(train_data)
   
   #Getting results for PHI1
    weights_phi1 = phi1_perceptron_train(train_data, cw_cl_counts, maxIter, scheme)
    flat_cor_phi1, flat_pre_phi1 = phi1_perceptron_test(test_data, weights_phi1, cw_cl_counts, scheme)

    print("\n ---------------------------------------------------------------------------")
    f1_micro = f1_score(flat_cor_phi1, flat_pre_phi1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    print('F1 Score for PHI 1: ', round(f1_micro, 5))
    print("--------------------------------------------------------------------------- \n")
   
    end = time.time()
    print("Total Time Elapsed: ", (end - start), " seconds\n")
if __name__ == '__main__':
    main()
