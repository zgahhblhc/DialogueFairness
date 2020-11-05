from tqdm import tqdm
import argparse
import importlib
import json

parser = argparse.ArgumentParser(description='Bias Detection')
parser.add_argument('--dialog_model', type=str, default=None, help='options: Seq2Seq, TransRanker')
parser.add_argument('--bias', type=str, default=None, help='options: gender, race')
parser.add_argument('--max_len', type=int, default=150)
args = parser.parse_args()
bias = args.bias

module = importlib.import_module("ParlAI" + args.dialog_model + "Twitter" + ".agent")
get_response = getattr(module, "get_response")

if bias == 'gender':
    with open('ParallelContextData/gender_corpus.json', 'r') as f:
        contexts = json.load(f)

elif bias == 'race':
    with open('ParallelContextData/race_corpus.json', 'r') as f:
        contexts = json.load(f)

f_left_out = open('results/' + args.dialog_model + '_' + bias + '_' + str(args.max_len) + '_left_results.txt', 'w')
f_right_out = open('results/' + args.dialog_model + '_' + bias + '_' + str(args.max_len) + '_right_results.txt', 'w')

batch_size = 100
n_batch = len(contexts) // batch_size

print("Inferring the responses to the contexts in the parallel context corpus...")
for i in tqdm(n_batch):
    context_batch = contexts[i * batch_size: (i + 1) * batch_size]
    left_context_batch = [c[0] for c in context_batch]
    right_context_batch = [c[1] for c in context_batch]
    left_response_batch = get_response(left_context_batch, args.max_len)
    right_response_batch = get_response(right_context_batch, args.max_len)

    for left_context, left_response, right_context, right_response in zip(left_context_batch, left_response_batch,
                                                                          right_context_batch,
                                                                          right_response_batch):

        f_left_out.write(left_context + '\t' + left_response + '\n')
        f_right_out.write(right_context + '\t' + right_response + '\n')