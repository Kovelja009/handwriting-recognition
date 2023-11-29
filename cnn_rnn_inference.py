import torch
from torch.autograd import Variable
from cnn_rnn_config import classes_, cdict_, icdict_, fixed_size
import editdistance
import argparse
from skimage import io as img_io
import numpy as np
from data.iamdataset import image_resize, centered
import cv2


# only works for 2 heads
# TODO: make it work only for RNN head
def inference(model, inputs, gpu_id=0):
    tdecs_rnn = []
    outputs = []
    for img in inputs:
        # add batch dimension
        img = img.unsqueeze(0)
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            output, aux_output = model(img)
        tdec = output.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs_rnn += [tdec]

    for tdec in tdecs_rnn:
        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict_[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()
        outputs += [dec_transcr]
    return outputs

def cer_wer(outputs, targets):
    cer, wer = [], []
    cntc, cntw = 0, 0
    for o, t in zip(outputs, targets):
        t = t.strip()
        # calculate CER and WER
        cc = float(editdistance.eval(o, t))
        ww = float(editdistance.eval(o.split(' '), t.split(' ')))
        cntc += len(t)
        cntw +=  len(t.split(' '))
        cer += [cc]
        wer += [ww]

    cer = sum(cer) / cntc
    wer = sum(wer) / cntw
    return cer, wer


def load_model(model_path='./saved_models/best.pth', gpu_id=0):
    model = torch.load(model_path)
    model = model.cuda(gpu_id)
    model.eval()
    return model

def preprocess_images(img_paths):
    imgs = []
    for img_path in img_paths:
        img = img_io.imread(img_path)
        img = 1 - img.astype(np.float32) / 255.0
        img = image_resize(img, height=img.shape[0] // 2)
        fheight, fwidth = fixed_size[0], fixed_size[1]
        nheight, nwidth = img.shape[0], img.shape[1]

        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))

        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))
        img = centered(img, (fheight, fwidth), border_value=0.0)
        img = torch.Tensor(img).float().unsqueeze(0)
        imgs += [img]
    return imgs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./saved_models/best.pth')
parser.add_argument('--image_paths', type=list, default=['./data/img_lines/a01-000u-00.png', './data/img_lines/a01-007u-00.png'], help='list of image paths')
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

model = load_model(args.model_path)
imgs = preprocess_images(args.image_paths)
outputs = inference(model, imgs, args.gpu_id)
for i, img in enumerate(imgs):
    print(f'Predicted: {outputs[i]}')
    cv2.imshow('image', img.numpy().squeeze())
    # wait for any key to exit
    cv2.waitKey(0)
