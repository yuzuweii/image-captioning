import nltk
import pickle
import argparse
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    allid = coco.anns.keys()
    for id in allid:
        caption = str(coco.anns[id]['caption']).lower()
        tokens = nltk.tokenize.word_tokenize(caption)
        counter.update(tokens)

    common = ['<pad>', '<start>', '<end>', '<unk>']
    vocab = Vocabulary()
    for w in common:
        vocab.add_word(w)

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    for word in words:
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='annotations/captions_train2014.json', 
                        help='path to annotation file for training')
    parser.add_argument('--vocab_path', type=str, default='./data2/vocab.pkl', 
                        help='path to save vocabulary')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)