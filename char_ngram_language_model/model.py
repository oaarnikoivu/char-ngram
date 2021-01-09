import re 
import math
import numpy as np 
from collections import defaultdict 
from itertools import product
from string import ascii_lowercase
from helpers import preprocess_line, read_distribution, extract_counts

# Vocabulary 
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ' ', '0', '#']

# List of all possible trigrams and bigrams 
possible_trigrams =  [a+b+c for a,b,c in product(alphabet, repeat=3)]
possible_bigrams = [a+b for a,b in product(alphabet, repeat=2)]

class TrigramLanguageModel:
    def __init__(self, infile, alpha=0.3, smoothing=False, lambdas=[], interpolate=False): 
        
        # Store trigram, bigram and unigram counts in separate dictionaries 
        self.trigram_counts  = extract_counts(infile, n_gram_range=3)
        self.bigram_counts  = extract_counts(infile, n_gram_range=2)
        self.unigram_counts  = extract_counts(infile, n_gram_range=1)

        # Estimated probabilities 
        self.train_probabilities = defaultdict(float)
        self.estimated_probabilities = defaultdict(float)

        self.total_types = len(self.unigram_counts) 
        self.total_tokens = sum(self.unigram_counts.values()) 
        
        # alpha parameter for smoothing 
        self.alpha = alpha 
        self.smoothing = smoothing 
        
        # lambda parameters for linear interpolation
        self.lambdas = lambdas
        self.interpolate = interpolate

    def get_unigram_probabilities(self, unigram):
        '''Calculate unigram probabilities using MLE'''
        numerator = self.unigram_counts.get(unigram, 0)
        denominator = self.total_tokens 
  
        if numerator == 0 or denominator == 0:
            return 0.0
        else:
            return float(numerator) / float(denominator)

    def get_bigram_probabilities(self, bigram):
        '''Calculate bigram probabilities using MLE'''
        numerator = self.bigram_counts.get(bigram, 0)
        denominator = self.unigram_counts.get(bigram[0], 0)

        if numerator == 0 or denominator == 0:
            return 0.0 
        else:
            return float(numerator) / float(denominator)

    def get_trigram_probabilities(self, trigram):
        '''Calculate trigram probabilities using MLE'''
        '''If smoothing enabled, calculate using add-alpha smoothing'''
        numerator = self.trigram_counts.get(trigram, 0)
        denominator = self.bigram_counts.get(trigram[:-1], 0)

        # If smoothing, add alpha to the numerator, and add alpha times the total types to the denominator 
        if self.smoothing:
            numerator += self.alpha
            denominator += (self.alpha * self.total_types)

        if numerator == 0 or denominator == 0:
            return 0.0
        else:
            return float(numerator) / float(denominator)

    def trigram_interpolate(self, trigram):
        '''Calculate trigram probabilities using interpolation'''
        assert np.around(np.sum(self.lambdas)) == 1.0, 'When using linear interploation, the lambda parameters must sum to 1!'
        c3, c2, c1 = trigram[2], trigram[1], trigram[0]
        unigram_probabilities = self.get_unigram_probabilities(c3)
        bigram_probabilities = self.get_bigram_probabilities(''.join([c2, c3]))
        trigram_probabilities =  self.get_trigram_probabilities(trigram)
        estimated_probabilities = self.lambdas[0] * unigram_probabilities + self.lambdas[1] * bigram_probabilities + self.lambdas[2] * trigram_probabilities
        return estimated_probabilities 

    def normalise_interpolate(self, estimated_probabilities): #added
        '''Normalises the interpolated probabilities of any trigram given a bigram to sum up to 1'''
        conditional_sums = {}
        for i in possible_bigrams:
            conditional_sums[i] = sum(estimated_probabilities[i+l] for l in alphabet)
        for key, value in estimated_probabilities.items():
            value = float(value)/float(conditional_sums[key[0:2]])
        return estimated_probabilities
    
    def write_probabilities_to_file(self, outfile):
        """
        outfile = destination file for estimated probabilities 
        - if interpolate is set to true, estimate the probabilities using linear interpolation 
        """

        # estimate probabilities and write to specified output file
        # if interpolate is set to True, estimate the probabilities using linear interpolation
        for trigram in self.trigram_counts.keys():
            if self.interpolate:
                self.estimated_probabilities[trigram] = self.trigram_interpolate(trigram)
            else:
                self.estimated_probabilities[trigram] = self.get_trigram_probabilities(trigram)
        if self.interpolate:
            self.estimated_probabilities = self.normalise_interpolate(self.estimated_probabilities)
        
        # write probabilites to outfile and sort by probability from largest to smallest 
        for p in sorted(self.estimated_probabilities.items(), key=lambda x: x[1], reverse=True):
            outfile.write(f'{p[0]}\t{p[1]}\n') 

    def get_probabilities(self):
        train_probabilities = {}
        for trigram in self.trigram_counts.keys():
            if self.interpolate:
                train_probabilities[trigram] = self.trigram_interpolate(trigram)
            else:
                train_probabilities[trigram] = self.get_trigram_probabilities(trigram)
        if self.interpolate:
            train_probabilities = self.normalise_interpolate(train_probabilities)

        return train_probabilities

    def perplexity(self, testfile, distribution):
        """
        testfile = location of test set 
        distribution = language model's estimated probability distribution 
        """
        log_probs = []

        # Extract trigrams from the test set, and append the log base 2 probabilities for each trigram which occurs in the trained 
        # lanugage model's probability distribution in the "log_probs" list 
        with open(testfile) as f:
            for line in f:
                line = preprocess_line(line)
                for j in range(len(line)-2):
                    trigram = line[j:j+3]
                    log_probs.append(np.log2(distribution[trigram]))
       
        # Compute entropy 
        entropy = (-1 / len(log_probs)) * np.sum(log_probs)

        # Compute perplexity 
        perplexity = math.pow(2.0, entropy)
        
        return entropy, perplexity