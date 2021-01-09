import re
import sys 
from model import TrigramLanguageModel
from helpers import create_from_lang_model, read_distribution

if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file
outfile = open("../data/out_probs", "w")
testfile = '../data/test'
distribution_file = '../data/out_probs'

#model = TrigramLanguageModel(infile, smoothing=True, alpha=0.1)

# uncomment to use model using linear interpolation
model = TrigramLanguageModel(infile, interpolate=True, lambdas=[0.0001, 0.0099, 0.99])

model.write_probabilities_to_file(outfile)
outfile.close()

distribution = read_distribution(distribution_file)

print('Generated sequence: ')
print(create_from_lang_model(distribution, N=300))

entropy, perplexity = model.perplexity(testfile, distribution)

print()
print(f'Entropy: {entropy}')
print(f'Perplexity: {perplexity}')