import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torchvision import transforms
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    transform = transforms.Compose([ 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    total_step = len(data_loader)
    train_losses = []
    train_acc = []
    for epoch in range(args.num_epochs):
        losses = []
        accuracy = 0.0

        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            losses.append(loss)
            _,topi = outputs.topk(1, dim=1)
            targets = targets.unsqueeze(-1)
            match = (topi == targets).sum()
            accuracy += float(match)/targets.shape[0]
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch {}, Step {}/{}, Loss: {:.4f}, Accuracy: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, i, total_step, loss.item(), accuracy/float(i+1)), np.exp(loss.item())) 
                
            if (i+1) % 1000 == 0:
                encoder_save_path = os.path.join(args.model_path, 'encoder-{}-{}-lemma.ckpt'.format(epoch+1, i+1))
                torch.save(encoder.state_dict(), encoder_save_path)
                decoder_save_path = os.path.join(args.model_path, 'decoder-{}-{}-lemma.ckpt'.format(epoch+1, i+1))
                torch.save(decoder.state_dict(), decoder_save_path)
                
        train_losses.append(sum(losses)/total_step)
        train_acc.append(accuracy/total_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='data2/models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='data2/vocab_lemma.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data2/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data2/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)