from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import cv2

class IAMDataset(Dataset):
    def __init__(self,
        data_path,
        img_height,
        img_width,
        subset = 'train'
        ):
        
        self.subset = subset
        self.data_path = data_path

        self.conH = img_height
        self.conW = img_width

        self.trainset_file = self.data_path + '/data/set_split/trainset.txt'
        self.valset_file = self.data_path + '/data/set_split/validationset1.txt'
        self.testset_file = self.data_path + '/data/set_split/testset.txt'
        self.line_file = self.data_path + '/data/ascii/lines.txt'
        self.word_file = self.data_path + '/data/ascii/words.txt'
        self.line_path = self.data_path + '/data/img_lines'

        self.finalize()

    def finalize(self):
        data = self.main_loader()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, text = self.data[index]
        image_norm = self.preprocess_image(img_path)

        image_norm = image_norm.reshape(1,self.conH,self.conW)
        sample = {'image': torch.from_numpy(image_norm), 'label': text}
        return sample

    def preprocess_image(self, img_path):
        image = cv2.imread(img_path, 0) #see channel and type

        h,w = image.shape

        imageN = np.ones((self.conH,self.conW))*255
        beginH = int(abs(self.conH-h)/2)
        beginW = int(abs(self.conW-w)/2)

        if h <= self.conH and w <= self.conW:
            imageN[beginH:beginH+h, beginW:beginW+w] = image
        elif float(h) / w > float(self.conH) / self.conW:
            newW = int(w * self.conH / float(h))
            beginW = int(abs(self.conW-newW)/2)
            image = cv2.resize(image, (newW, self.conH))
            imageN[:,beginW:beginW+newW] = image
        elif float(h) / w <= float(self.conH) / self.conW:
            newH = int(h * self.conW / float(w))
            beginH = int(abs(self.conH-newH)/2)
            image = cv2.resize(image, (self.conW, newH))
            imageN[beginH:beginH+newH] = image

        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/12

        return imageN

    def main_loader(self) -> list:
        if self.subset == 'train':
            valid_set = np.loadtxt(self.data_path + '/data/aachen_iam_split/train.uttlist', dtype=str)
            #print(valid_set)
        elif self.subset == 'val':
            valid_set = np.loadtxt(self.data_path + '/data/aachen_iam_split/validation.uttlist', dtype=str)
        elif self.subset == 'test':
            #valid_set = np.loadtxt(self.testset_file, dtype=str)
            valid_set = np.loadtxt(self.data_path + '/data/aachen_iam_split/test.uttlist', dtype=str)
        else:
            raise ValueError

        gt = []
        for line in open(self.line_file):
            if not line.startswith("#"):
                info = line.strip().split()
                name = info[0]
                pathlist = [self.line_path] + [name]
                line_name = pathlist[-1]
                form_name = '-'.join(line_name.split('-')[:-1])


                if (form_name not in valid_set):
                    #print(line_name)
                    continue
                img_path = '/'.join(pathlist)
                img_path += '.png'
                transcr = ' '.join(info[8:])

                            # transform iam transcriptions
                transcr = transcr.replace(" ", "")
                # "We 'll" -> "We'll"
                special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
                # lower-case
                for cc in special_cases:
                    transcr = transcr.replace("|\'" + cc, "\'" + cc)
                    transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

                # dont replace if after | is stopword
                signs = "!.,:%)?"
                modified_text = ""
                for i, char in enumerate(transcr):
                    if char == "|" and (i + 1 < len(transcr) and transcr[i + 1] not in signs):
                        modified_text += " "
                    elif char != "|":
                        modified_text += char
                transcr = modified_text

                gt.append((img_path, transcr))
        return gt