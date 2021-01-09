import re
import random 
from model import TrigramLanguageModel
from helpers import read_distribution

def create_train_dev_sets(filename, train_out, dev_out):
    """
    filename = file to split 
    train_out = location of new train file
    dev_out = location of validation file 
    """
    with open(filename) as f:
        lines = f.readlines() 
    
    # shuffle lines randomly 
    random.shuffle(lines)

    # do 90:10% split 
    numlines = int(len(lines)*0.1)
    
    # assign 10% of lines to validation set
    with open(dev_out, 'w') as f:
        f.writelines(lines[:numlines])

    # assign 90% of lines to training set 
    with open(train_out, 'w') as f:
        f.writelines(lines[numlines:])

def find_optimal_alpha(train, dev):
    """
    train = training set 
    dev = validation set
    """
    # smoothing parameters 
    alphas = [0.0000001, 0.000001, 0.0001, 0.0001, 0.01, 0.1, 1]

    # store computes perplexy values 
    perplexities = []
    
    print('Finding optimal alpha, this may take a while...')

    # For each alpha, create a new Trigram model and compute perplexity on the validation set
    for alpha in alphas:
        model = TrigramLanguageModel(train, smoothing=True, alpha=alpha)
        distribution = model.get_probabilities()
        _, perplexity = model.perplexity(dev, distribution)
        perplexities.append(perplexity)
    
    optimal_alpha = 0
    optimal_perplexity = min(perplexities)

    # Retrieve alpha which resulted in the lowest perplexity score 
    for a, p in zip(alphas, perplexities):
        if p == min(perplexities):
            optimal_alpha = a

    print(f'Optimal alpha: {optimal_alpha}\n Perplexity: {optimal_perplexity}')

def find_optimal_lambdas(train, dev):
    """
    train = training set 
    dev = validation set
    """
    # Lambda parameters 
    lambdas = [[0.2, 0.2, 0.6], [0.1, 0.2, 0.7], [0.01, 0.09, 0.9], [0.001, 0.1, 0.899], [0.001, 0.099, 0.9], [0.0001, 0.0099, 0.99], [0.1, 0.1, 0.8]]

    # Store optimal lambdas 
    optimal_lambdas = []

    # Store computed perplexity values 
    perplexities = []

    print('Finding optimal lambdas, this may take a while...')

    # For each lambda, create a new Trigram model and compute perplexity on the validation set
    for l in lambdas:
        model = TrigramLanguageModel(train, interpolate=True, lambdas=l)
        distribution = model.get_probabilities()
        _, perplexity = model.perplexity(dev, distribution)
        perplexities.append(perplexity)
    

    optimal_perplexity = min(perplexities)

    # Retrieve alpha which resulted in the lowest perplexity score 
    for l, p in zip(lambdas, perplexities):
        if p == min(perplexities):
            optimal_lambdas.append(l)

    print(f'Optimal lambdas: {optimal_lambdas}\n Perplexity: {optimal_perplexity}')

# create train dev split for english
en_file = '../data/training.en'
en_train_out = '../data/train_dev/train.en'
en_dev_out = '../data/train_dev/dev.en'

# uncomment to create new training and validation set for english 
#create_train_dev_sets(en_file, en_train_out, en_dev_out)

find_optimal_alpha(train=en_train_out, dev=en_dev_out)
find_optimal_lambdas(train=en_train_out, dev=en_dev_out)