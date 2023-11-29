import io,os
import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
# from auxilary_functions import image_resize, centered
import tqdm

class IAMDataset(Dataset):
    def __init__(self,
        fixed_size,
        transforms = None, # List of augmentation transform functions to be applied on each input
        character_classes = None, # If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        subset = 'train'
        ):

        self.transforms = transforms
        self.character_classes = character_classes
        self.subset = subset
        self.fixed_size = fixed_size

        self.trainset_file = './data/set_split/trainset.txt'
        self.valset_file = './data/set_split/validationset1.txt'
        self.testset_file = './data/set_split/testset.txt'
        self.line_file = './data/ascii/lines.txt'
        self.word_file = './data/ascii/words.txt'
        self.line_path = './data/img_lines'
        self.finalize()

    def finalize(self):
        
        save_file = './data/saved_datasets/{}_IAM.pt'.format(self.subset) #dataset_path + '/' + set + '_IAM.pt'
        if isfile(save_file) is False:
            data = self.main_loader()
            torch.save(data, save_file)   # Uncomment this in 'release' version
        else:
            data = torch.load(save_file)
        self.data = data

        if self.character_classes is None:
            res = set()
            # compute character classes given input transcriptions
            for _,transcr in tqdm.tqdm(data):
                res.update(list(transcr))
            res = sorted(list(res))
            self.character_classes = res

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index][0]
        transcr = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        if self.subset == 'train':
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]

        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))

        img = centered(img, (fheight, fwidth), border_value=0.0)
        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    img = tr(img)

        img = torch.Tensor(img).float().unsqueeze(0)
        return img, transcr
    

    def main_loader(self) -> list:
        def gather_iam_info(self):
            if self.subset == 'train':
                valid_set = np.loadtxt('./data/aachen_iam_split/train.uttlist', dtype=str)
                #print(valid_set)
            elif self.subset == 'val':
                valid_set = np.loadtxt('./data/aachen_iam_split/validation.uttlist', dtype=str)
            elif self.subset == 'test':
                #valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt('./data/aachen_iam_split/test.uttlist', dtype=str)
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
                    transcr = ' '.join(info[8:])
                    gt.append((img_path, transcr))
            return gt
        
        info = gather_iam_info(self)
        data = []
        for i, (img_path, transcr) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            

            try:
                img = img_io.imread(img_path + '.png')
                img = 1 - img.astype(np.float32) / 255.0
                img = image_resize(img, height=img.shape[0] // 2)
            except:
               print('Could not add image file {}.png'.format(img_path))
               continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")

            data += [(img, transcr)]
        return data

    
    
    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        '''
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not
        
        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        '''
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError('The requested fixed image size is invalid!')
            new_img = resize(image=img, output_shape=fixed_image_size[::-1], mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print('OUCH')
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img
        

def affine_transformation(img, m=1.0, s=.2, border_value=None):
    h, w = img.shape[0], img.shape[1]
    src_point = np.float32([[w / 2.0, h / 3.0],
                            [2 * w / 3.0, 2 * h / 3.0],
                            [w / 3.0, 2 * h / 3.0]])
    random_shift = m + np.random.uniform(-1.0, 1.0, size=(3,2)) * s
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if border_value is None:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(w, h), borderValue=float(border_value))
    return warped_img

def image_resize(img, height=None, width=None):

    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img


def centered(word_img, tsize, centering=(.5, .5), border_value=None):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)
    return word_img