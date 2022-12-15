import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import model
import wandb
from torch.autograd import Variable

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

parser.add_argument('--log-file', type=str,  default='',
                    help='path to save the log')
parser.add_argument('--mmd_kernel_alpha', type=float,  default=0.5,
                    help='mmd kernel')
parser.add_argument('--mmd_lambda', type=float,  default=0.2,
                    help='mmd kernel')
parser.add_argument('--moment', action='store_true',
                    help='using moment regularization')
parser.add_argument('--moment_split', type=int, default=1000,
                    help='threshold for rare and popular words')
parser.add_argument('--moment_lambda', type=float, default=0.02,
                    help='lambda')
parser.add_argument('--adv', action='store_false',
                    help='using adversarial regularization')
parser.add_argument('--adv_bias', type=int, default=1000,
                    help='threshold for rare and popular words')
parser.add_argument('--adv_lambda', type=int, default=0.02,
                    help='lambda')
parser.add_argument('--adv_lr', type=float,  default=0.02,
                    help='adv learning rate')
parser.add_argument('--adv_wdecay', type=float,  default=1.2e-6,
                    help='adv weight decay')

parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id(),
                    help='wandb run id for resuming')

args = parser.parse_args()
args.tied = True

# Set to true, if you want ot use WandB
wandb_usage = False
# Initialize WandB logging
if wandb_usage:
    if args.resume == "":
        wandb.init(
            project="FRAGE-LSTM-news-db-v2",
            entity="<YOUR_USERNAME>",
            config=args,
            id=args.wandb_id)
    else:
        wandb.init(
            project="FRAGE-LSTM-news-db-v2",
            entity="<YOUR_USERNAME>",
            config=args,
            resume="must",
            id=args.wandb_id)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, epochs_trained], f)

def model_load(fn):
    global model, criterion, optimizer, epochs_trained
    with open(fn, 'rb') as f:
        model, criterion, optimizer, epochs_trained = torch.load(f)

import os
import hashlib
fn = os.path.join(
    args.data, 'frage-corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest()))
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

print("Dictionary size:", len(corpus.dictionary))


###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None
ntokens = len(corpus.dictionary)
if args.adv:
   rate = (ntokens - args.adv_bias) * 1.0 / ntokens
   adv_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([rate, 1 - rate]).cuda())
   adv_hidden = nn.Linear(args.emsize, 2).cuda()
   adv_targets = torch.LongTensor(np.array([0] * args.adv_bias + [1] * (ntokens - args.adv_bias))).cuda()
   adv_targets = Variable(adv_targets)
   adv_hidden.weight.data.uniform_(-0.1, 0.1)
ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
epochs_trained = 0
###
if args.resume:
    model_load(args.resume)
    print(f'Resuming model from epoch {epochs_trained} ...')
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

if wandb_usage:
    wandb.watch(model, log_freq=100)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        # Fix according to https://stackoverflow.com/questions/56483122/indexerror-invalid-index-of-a-0-dim-tensor-use-tensor-item-to-convert-a-0-di
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(epoch):
    inner_product = 0
    count = 0
    save_hiddens = []
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets, noise_weight=None, noise=False)
        if args.moment:
            bias = args.moment_split
            common = model.encoder.weight[:bias]
            rare = model.encoder.weight[bias:]
            mean0 = torch.mean(common, 0)
            mean1 = torch.mean(rare, 0)
            var0 = torch.var(common, 0)
            var1 = torch.var(rare, 0)
            kewness0 = torch.mean(torch.pow(common - mean0, 3), 0) / torch.pow(var0, 1.5)
            kewness1 = torch.mean(torch.pow(rare - mean1, 3), 0) / torch.pow(var1, 1.5)
            kurtosis0 = torch.mean(torch.pow(common - mean0, 4), 0) / torch.pow(var0, 2)
            kurtosis1 = torch.mean(torch.pow(rare - mean1, 4), 0) / torch.pow(var1, 2)
            reg_loss = torch.sqrt(torch.sum(torch.pow(mean0 - mean1, 2))) + torch.sqrt(torch.sum(torch.pow(var0 - var1, 2))) \
                      + torch.sqrt(torch.sum(torch.pow(kewness0 - kewness1, 2))) + torch.sqrt(torch.sum(torch.pow(kurtosis0 - kurtosis1, 2)))
            loss = raw_loss + args.moment_lambda * reg_loss
        elif args.adv:
           # calculate the adv_classifier
            optimizer.zero_grad()
            adv_optimizer.zero_grad()
            adv_h = adv_hidden(model.encoder.weight)
            adv_loss = adv_criterion(adv_h, adv_targets)
            adv_loss.backward()
            adv_optimizer.step()

            hidden = repackage_hidden(hidden)
            adv_optimizer.zero_grad()
            optimizer.zero_grad()
            #output, hidden, rnn_hs, dropped_rnn_hs, w = model(data, hidden, return_h=True)
            #raw_loss = criterion(output.view(-1, ntokens), targets)

            adv_h = adv_hidden(model.encoder.weight[args.adv_bias:])
            adv_loss = adv_criterion(adv_h, adv_targets[args.adv_bias:])
            loss = raw_loss - args.adv_lambda * adv_loss
        else:
            loss = raw_loss
        #loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
            if wandb_usage:
                wandb.log({"batch_loss": cur_loss, "epoch": epoch})
        ###
        batch += 1
        i += seq_len
        #if i >= 30 and i <= 150:
            #import pickle
            #with open('hiddens', 'wb') as f:
            #    pickle.dump(save_hiddens, f)
            #print('OK!!')
    #print('inner_product', inner_product / count)
# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
finetune = False
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    if args.adv:
        adv_optimizer = torch.optim.SGD(adv_hidden.parameters(), lr=args.adv_lr, weight_decay=args.adv_wdecay)
    #optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    # for epoch in range(1, args.epochs+1):
    while epochs_trained < args.epochs:
        epoch_start_time = time.time()
        epoch = epochs_trained + 1
        train(epoch)
        epochs_trained += 1

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                try:
                    prm.data = optimizer.state[prm]['ax'].clone()
                except:
                    pass
            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if epoch % 30 == 0:
                test_loss = evaluate(test_data, test_batch_size)
                print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                      'test ppl {:8.2f} | test bpc {:8.3f}'.format(
                   epoch, (time.time() - epoch_start_time), test_loss, math.exp(test_loss), test_loss / math.log(2)))
                print('=' * 89)
                if wandb_usage:
                    wandb.log({"test_loss": test_loss, "epoch": epoch})

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            #if epoch == 1800:
            #    finetune = True
            #    print('Switching!')
            #    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            #if epoch == 1000:
            if (not finetune and epoch == 1000) or (not finetune and (len(best_val_loss)>args.nonmono and val_loss2 > min(best_val_loss[:-args.nonmono]))):
                finetune = True
                print('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            best_val_loss.append(val_loss2)
        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            if wandb_usage:
                wandb.log({"valid_loss": val_loss, "epoch": epoch})

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if epoch % 30 == 0:
                test_loss = evaluate(test_data, test_batch_size)
                print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                      'test ppl {:8.2f} | test bpc {:8.3f}'.format(
                   epoch, (time.time() - epoch_start_time), test_loss, math.exp(test_loss), test_loss / math.log(2)))
                print('=' * 89)
                if wandb_usage:
                    wandb.log({"test_loss": test_loss, "epoch": epoch})

            if epoch >= 110:
            #if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

        if epoch % 10 == 0:
            print("Saving regular model checkpoint.")
            model_save('{}.e{}'.format(args.save, epoch))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
if wandb_usage:
    wandb.log({"final_loss": test_loss})
