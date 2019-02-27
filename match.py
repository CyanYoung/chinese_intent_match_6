import pickle as pk

import torch

from represent import sent2ind

from preprocess import clean

from util import map_item


device = torch.device('cpu')

seq_len = 30

path_word_ind = 'feat/word_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

paths = {'esi': 'model/rnn_esi.pkl'}

models = {'esi': torch.load(map_item('esi', paths), map_location=device)}


def predict(text1, text2, name):
    text1, text2 = clean(text1), clean(text2)
    pad_seq1 = sent2ind(text1, word_inds, seq_len, keep_oov=True)
    pad_seq2 = sent2ind(text2, word_inds, seq_len, keep_oov=True)
    sent1 = torch.LongTensor([pad_seq1]).to(device)
    sent2 = torch.LongTensor([pad_seq2]).to(device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        prob = torch.sigmoid(model(sent1, sent2))[0][0]
    return '{:.3f}'.format(prob)


if __name__ == '__main__':
    while True:
        text1, text2 = input('text1: '), input('text2: ')
        print('esi: %s' % predict(text1, text2, 'esi'))
