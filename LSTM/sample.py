import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle
import os
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from PIL import Image
import pdb

class Vocabulary(object):
    """Simple vocabulary wrapper."""
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu'))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    #sampled_ids = decoder.sample(feature)
    #sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # sample by beam search
    sampled_ids = decoder.sample_beam_search(feature)
    num_sents = min(len(sampled_ids), 3)
    
    sentences = []
    for sampled_id in sampled_ids[:num_sents]:
    #Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_id:
            word = vocab.idx2word[int(word_id)]
            sampled_caption.append(word)
            if word == '<end>':
                sentence = ' '.join(sampled_caption)
                sentences.append(sentence)
                break
            
    for sentence in list(set(sentences)): #print only the unique sentences
        print(sentence)
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))

    return list(set(sentences))

def run(imgDir, encoder_path, decoder_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default= imgDir, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default=encoder_path, help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default=decoder_path, help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data2/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    return main(args)

if __name__ == '__main__':
    run('/Users/yuzuwei/Downloads/code/LSTM/test-image.jpg', '/Users/yuzuwei/Downloads/code/LSTM/data2/models/LSTM256_encoder_epoch10.ckpt', '/Users/yuzuwei/Downloads/code/LSTM/data2/models/LSTM256_decoder_epoch10.ckpt')