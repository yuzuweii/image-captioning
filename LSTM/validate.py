import torch
import argparse
import pickle 
from data_loader import get_loader 
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import sentence_bleu
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sb1,sb2,sb3,sb4=0,0,0,0
def get_bleu(lengths, vocab, sb1, sb2, sb3, sb4, topi, targets):
    sentence_len=0
    for j in range(len(lengths)):
        topi_sen = topi[sentence_len:sentence_len+lengths[j]]
        candidate = [vocab.idx2word[int(idx[0])] for idx in topi_sen]
        target_sen = targets[sentence_len:sentence_len+lengths[j]]
        reference = [[vocab.idx2word[int(idx[0])] for idx in target_sen]]

        sb1+=float(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))/len(lengths)
        sb2+=float(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))/len(lengths)
        sb3+=float(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))/len(lengths)
        sb4+=float(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))/len(lengths)
        sentence_len+=lengths[j]
    return sb1, sb2, sb3, sb4

def val(args):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    encoder = EncoderCNN(args.embed_size).eval() 
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    criterion = nn.CrossEntropyLoss()
    
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    print('val_loader length = {}'.format(len(data_loader)))
    
    val_accuracy_all = 0
    val_loss_all = 0
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            val_loss_all += criterion(outputs, targets)
            
            _,topi = outputs.topk(1, dim=1)
            targets = targets.unsqueeze(-1)
            val_accuracy_all += float((topi == targets).sum())/targets.shape[0]

            if i % 100 == 0:
                print('step {}/{}, accuracy {}, bleu_score {}{}{}{}'.format(i, len(data_loader), val_accuracy_all/(i+1), sb1/(i+1),sb2/(i+1),sb3/(i+1),sb4/(i+1)))   
            sb1, sb2, sb3, sb4 = get_bleu(lengths, vocab, sb1, sb2, sb3, sb4, topi, targets)

    val_loss = val_loss_all/len(data_loader)
    print('loss: {:.4f}'.format(val_loss))
    print('accuracy: {:.4f}',val_accuracy_all/len(data_loader))

    print('Cumulative 1-gram: %f' % (sb1/len(data_loader)))
    print('Cumulative 2-gram: %f' % (sb2/len(data_loader)))
    print('Cumulative 3-gram: %f' % (sb3/len(data_loader)))
    print('Cumulative 4-gram: %f' % (sb4/len(data_loader)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default= '/data2/models/encoder-3-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default= '/data2/models/decoder-3-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default= '/data2/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default= '/data2/val_resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default= '/data2/annotations/captions_val2014.json', help='path for train annotation json file')
    parser.add_argument('--embed_size', type=int , default=128, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    print(args)
    val(args)
    
    
    
    
    
    
    
    
    
   
    