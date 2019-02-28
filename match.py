import pickle as pk

import torch

from represent import sent2ind

from preprocess import clean

from util import map_item


device = torch.device('cpu')

seq_len = 30

bos, sep = '<', '-'

path_word_ind = 'feat/word_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

paths = {'trm': 'model/dnn_trm.pkl'}

models = {'trm': torch.load(map_item('trm', paths), map_location=device)}


def predict(text1, text2, name):
    text1, text2 = clean(text1), clean(text2)
    text = bos + text1 + sep + text2
    pad_seq = sent2ind(text, word_inds, seq_len * 2, keep_oov=True)
    sent = torch.LongTensor([pad_seq]).to(device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        prob = torch.sigmoid(model(sent))[0][0]
    return '{:.3f}'.format(prob)


if __name__ == '__main__':
    while True:
        text1, text2 = input('text1: '), input('text2: ')
        print('trm: %s' % predict(text1, text2, 'trm'))
