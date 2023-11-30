import argparse
import logging
import matplotlib.pyplot as plt

import os
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


from data.iamdataset import IAMDataset

from cnn_rnn_config import *

from cnn_rnn_models import CNN_RNN

from data.auxilary_functions import affine_transformation

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter



logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('CNN_RNN-Experiment::train')
logger.info('--- Running CNN_RNN Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                    help='lr')
parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                    help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
parser.add_argument('--display', action='store', type=int, default=100,
                    help='The number of iterations after which to display the loss values. Default: 100')
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--scheduler', action='store', type=str, default='mstep')
parser.add_argument('--remove_spaces', action='store_true')
parser.add_argument('--resize', action='store_true')
parser.add_argument('--head_layers', action='store', type=int, default=3)
parser.add_argument('--head_type', action='store', type=str, default='rnn')

args = parser.parse_args()

gpu_id = args.gpu_id
logger.info('###########################################')

# prepare dataset loader

logger.info('Loading dataset.')

aug_transforms = [lambda x: affine_transformation(x, s=.1)]


train_set = IAMDataset(subset='train', fixed_size=fixed_size, transforms=aug_transforms)
classes = train_set.character_classes
print('# training lines ' + str(train_set.__len__()))

val_set = IAMDataset(subset='val', fixed_size=fixed_size, transforms=None)
print('# validation lines ' + str(val_set.__len__()))

test_set = IAMDataset(subset='test', fixed_size=fixed_size, transforms=None)
print('# testing lines ' + str(test_set.__len__()))

classes = '_' + ''.join(classes)

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
if val_set is not None:
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

# load CNN
logger.info('Preparing Net...')

# get me the name of gpu from gpu_id
if gpu_id >= 0 and torch.cuda.is_available():
    print(f'Using: {torch.cuda.get_device_name(gpu_id)}')
else:
    print('Using: CPU')


if args.head_layers > 0:
    head_cfg = (head_cfg[0], args.head_layers)

head_type = args.head_type

if load_model:
    net = torch.load(save_path + 'best_rnn_head.pth')
else:
    net = CNN_RNN(cnn_cfg, head_cfg, len(classes), head=head_type, flattening=flattening, stn=stn)
net.cuda(args.gpu_id)

ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='sum', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt) /batch_size

#ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='mean', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt)

#restart_epochs = 40 #max_epochs // 6

nlr = args.learning_rate

parameters = list(net.parameters())
optimizer = torch.optim.AdamW(parameters, nlr, weight_decay=0.00005)

if 'mstep' in args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])
elif 'cos' in args.scheduler:
    restart_epochs = int(args.scheduler.replace('cos', ''))
    if not isinstance(restart_epochs, int):
        print('define restart epochs as cos40')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)
else:
    print('not supported scheduler! choose eithe mstep or cos')


def train(epoch):

    net.train()
    closs = []
    cumulative_loss = []
    for iter_idx, (img, transcr) in enumerate(train_loader):
        optimizer.zero_grad()

        img = Variable(img.cuda(gpu_id))

        with torch.no_grad():
            rids = torch.BoolTensor(torch.bernoulli(.33 * torch.ones(img.size(0))).bool())
            if sum(rids) > 1 :
                img[rids] += (torch.rand(img[rids].size(0)).view(-1, 1, 1, 1) * 1.0 * torch.randn(img[rids].size())).to(img.device)

        img = img.clamp(0,1)

        if head_type == "both":
            output, aux_output = net(img)
        else:
            output = net(img)

        act_lens = torch.IntTensor(img.size(0)*[output.size(0)])
        labels = torch.IntTensor([cdict[c] for c in ''.join(transcr)])
        label_lens = torch.IntTensor([len(t) for t in transcr])

        loss_val = ctc_loss(output.cpu(), labels, act_lens, label_lens)
        closs += [loss_val.item()]

        if head_type == "both":
            loss_val += 0.1 * ctc_loss(aux_output.cpu(), labels, act_lens, label_lens)
      

        loss_val.backward()

        optimizer.step()


        # mean runing errors??
        if iter_idx % args.display == args.display-1:
            train_loss = sum(closs)/len(closs)
            cumulative_loss.append(train_loss)
            logger.info('Epoch: %d, Iteration: %d, train loss-> %f', epoch, iter_idx+1, train_loss)
             # Log training loss
            writer.add_scalar('Training Loss', train_loss, epoch * len(train_loader) + iter_idx)

            
            closs = []

            net.eval()

            tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
            # # show test image
            # plt.imshow(tst_img.squeeze().numpy(), cmap='gray')
            # plt.show()
            print('orig:: ' + tst_transcr)
            with torch.no_grad():
                timg = Variable(tst_img.cuda(gpu_id)).unsqueeze(0)

                tst_o = net(timg)
                if head_type == 'both':
                    tst_o = tst_o[0]

                tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
                tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
                print('gdec:: ' + ''.join([icdict[t] for t in tt]).replace('_', ''))

            net.train()

    if len(closs) > 0:
        logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))

    return sum(cumulative_loss)/len(cumulative_loss)


import editdistance
# slow implementation
def test(epoch, tset='test'):
    net.eval()

    if tset=='test':
        loader = test_loader
    elif tset=='val':
        loader = val_loader
    else:
        print("not recognized set in test function")

    logger.info('Testing ' + tset + ' set at epoch %d', epoch)

    tdecs = []
    transcrs = []
    for (img, transcr) in loader:
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o = net(img)
        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs += [tdec]
        transcrs += list(transcr)

    tdecs = np.concatenate(tdecs)

    cer, wer = [], []
    cntc, cntw = 0, 0
    for tdec, transcr in zip(tdecs, transcrs):
        transcr = transcr.strip()
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()

        # calculate CER and WER
        cc = float(editdistance.eval(dec_transcr, transcr))
        ww = float(editdistance.eval(dec_transcr.split(' '), transcr.split(' ')))
        cntc += len(transcr)
        cntw +=  len(transcr.split(' '))
        cer += [cc]
        wer += [ww]

    cer = sum(cer) / cntc
    wer = sum(wer) / cntw

    logger.info('CER at epoch %d: %f', epoch, cer)
    logger.info('WER at epoch %d: %f', epoch, wer)

     # Log CER and WER
    writer.add_scalar(f'Character Error Rate (CER): {tset}', cer, epoch)
    writer.add_scalar(f'Word Error Rate (WER): {tset}', wer, epoch)


    net.train()


# should use this one as in original paper
def test_both(epoch, tset='test'):
    net.eval()

    if tset=='test':
        loader = test_loader
    elif tset=='val':
        loader = val_loader
    else:
        print("not recognized set in test function")

    logger.info('Testing ' + tset + ' set at epoch %d', epoch)

    tdecs_rnn = []
    tdecs_cnn = []
    transcrs = []
    for (img, transcr) in loader:
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            o, aux_o = net(img)

        tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs_rnn += [tdec]

        tdec = aux_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs_cnn += [tdec]


        transcrs += list(transcr)

    cases = ['rnn', 'cnn']
    tdecs_list = [np.concatenate(tdecs_rnn), np.concatenate(tdecs_cnn)]
    for case, tdecs in zip(cases, tdecs_list):
        logger.info('Case: %s', case)
        cer, wer = [], []
        cntc, cntw = 0, 0
        for tdec, transcr in zip(tdecs, transcrs):
            transcr = transcr.strip()
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            dec_transcr = dec_transcr.strip()

            # calculate CER and WER
            cc = float(editdistance.eval(dec_transcr, transcr))
            ww = float(editdistance.eval(dec_transcr.split(' '), transcr.split(' ')))
            cntc += len(transcr)
            cntw +=  len(transcr.split(' '))
            cer += [cc]
            wer += [ww]

        cer = sum(cer) / cntc
        wer = sum(wer) / cntw

        logger.info('CER at epoch %d: %f', epoch, cer)
        logger.info('WER at epoch %d: %f', epoch, wer)

        # Log CER and WER
        writer.add_scalar(f'Character Error Rate (CER): {tset} - {case}', cer, epoch)
        writer.add_scalar(f'Word Error Rate (WER): {tset} - {case}', wer, epoch)

    net.train()

cnt = 1
logger.info('Training:')

# Initialize TensorBoard writer
writer = SummaryWriter()

best_loss = 10000000000
early_stop_counter = 0
should_stop = False
cum_loss = 10000000000

for epoch in range(1, max_epochs + 1):
    cum_loss = train(epoch)
    scheduler.step()

    if epoch % 10 == 0:
        if head_type=="both":
            if val_set is not None:
                test_both(epoch, 'val')
        else:
            if val_set is not None:
                test(epoch, 'val')

    if cum_loss < best_loss:
        best_loss = cum_loss
        early_stop_counter = 0
        logger.info('Saving net after %d epochs', epoch)
        torch.save(net.cpu(), save_path + 'best_rnn_head.pth')
        net.cuda(args.gpu_id)       
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stopping:
            should_stop = True

    if should_stop:
        print(f'Early stopping at epoch {epoch}')
        break


    if 'cos' in args.scheduler:
        if epoch % restart_epochs == 0:
            parameters = list(net.parameters())
            optimizer = torch.optim.AdamW(parameters, nlr, weight_decay=0.00005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)

writer.close()