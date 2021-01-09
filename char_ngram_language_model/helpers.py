import re
import numpy as np
from collections import defaultdict
from itertools import product

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ' ', '0', '#']

possible_trigrams =  [a+b+c for a,b,c in product(alphabet, repeat=3)]
possible_bigrams = [a+b for a,b in product(alphabet, repeat=2)]

def preprocess_line(str_line):
        processed_chars = ['##']
        for char in str_line:
            permitted = re.search("[a-zA-Z0-9. ]", char)
            if permitted:
                if char.isnumeric():
                    char = "0"
                processed_chars.append(char)
        processed_line = "".join(processed_chars)
        return processed_line.lower()

def get_counts(infile, n_gram_range):
    n_gram_counts = defaultdict(int)

    # if n_gram_range is 1, extract unigrams 
    # else if n_gram is 2, extract bigrams
    # otheriwse extract trigrams 
    if n_gram_range == 1:
        for key in alphabet:
            n_gram_counts[key] = 0
    elif n_gram_range == 2:
        for key in possible_bigrams:
            n_gram_counts[key] = 0
    else:
        for key in possible_trigrams:
            n_gram_counts[key] = 0
     
    with open(infile) as f:
        for line in f:
            # apply preprocessing to line 
            line = preprocess_line(line)
            for j in range(len(line)-(n_gram_range - 1)):
                # add n_grams to n_gram_count dictionary 
                n_gram = line[j:j+n_gram_range]
                n_gram_counts[n_gram] += 1
    return n_gram_counts

def create_from_lang_model(distribution, N):
    # alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ' ', '0']

    output = []
    c1, c2 = " ", " "

    for i in range(N):
        options_trigrams = []
        choice = []
        for char in alphabet:
            options_trigrams.append(c1+c2+char) #all trigrams of form (character of alphabet given bigram c1 c2)
        
        values = list(distribution[m] for m in options_trigrams) #probabilities for options
        
        if np.sum(values) != 1: #correct rounding error if necessary
            values /= np.sum(values)

        choice = np.random.choice(options_trigrams, p = values)
        c1, c2 = choice[1], choice[2]
        output.append(c2)
    return "".join(output)    

def read_distribution(filename):
    distribution = {}
    with open(filename, "r") as f:
        for line in f:
            s = re.split(r'\t+', line.rstrip()) # split by tab character and remove unwanted characters
            distribution[s[0]] = float(s[1])
        return distribution