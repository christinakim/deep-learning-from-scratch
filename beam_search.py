import numpy as np
import torch
from models import InferSent
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = 'fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)

sentences = ["I am an engineer now.",
             "You can be an engineer.",
             "Building stuff is very fun.",
             "Stuff breaks often too though."]
infersent.build_vocab(sentences, tokenize=True)

embeddings = infersent.encode(sentences, tokenize=True)

infersent.visualize('A man plays an instrument.', tokenize=True)


encoded_sentences = embeddings

# greedy decoder
def greedy_decoder(data):
    # index for largest probability each row
    return [np.argmax(s) for s in data]


# decode sequence
result = greedy_decoder(encoded_sentences)
print(result)


# beam search
def beam_search_decoder(encoded_data, beam_width: int):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in encoded_data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                # why are we taking the log of the product of the probabilities
                # instead of just the product of the probabilities?
                # the probabilities are all numbers less than 1,
                # multiplying a lot of numbers less than 1 will result in a very smol number
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select beam_width best
        sequences = ordered[:beam_width]
    return sequences


result = beam_search_decoder(encoded_sentences, 3)
# print result
for seq in result:
    print(seq)


# beam search with a width of 1 is equivalent to greedy search
assert beam_search_decoder(encoded_sentences, 1) == greedy_decoder(encoded_sentences)
