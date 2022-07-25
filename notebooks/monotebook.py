#!/usr/bin/env python
# coding: utf-8

# [1] build_nocall_detector.ipynb:

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install timm')

import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import warnings 
warnings.filterwarnings('ignore')

OUTPUT_DIR = '../output/10_output_dir/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class CFG:
    print_freq=100
    num_workers=4
    model_name= 'resnext50_32x4d'
    dim=(128, 281)
    scheduler='CosineAnnealingWarmRestarts'
    epochs=10
    lr=1e-4
    T_0=10 # for CosineAnnealingWarmRestarts
    min_lr=5e-7 # for CosineAnnealingWarmRestarts
    batch_size=32
    weight_decay=1e-6
    max_grad_norm=1000
    seed=42
    target_size=2
    target_col='hasbird'
    n_fold = 5
    pretrained = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv('../input/ff1010bird-duration7/rich_metadata.csv')
train.loc[train['hasbird']==0, 'filepath'] = '../input/ff1010bird-duration7/nocall/' + train.query('hasbird==0')['filename'] + '.npy'
train.loc[train['hasbird']==1, 'filepath'] = '../input/ff1010bird-duration7/bird/' + train.query('hasbird==1')['filename'] + '.npy'

train = train.dropna().reset_index(drop=True)

folds = train.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold', CFG.target_col]).size())

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df['filepath'].values
        self.labels = df['hasbird'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_paths[idx]
        file_path = file_name
        image = np.load(file_path)
        image = image.transpose(1,2,0)
        image = np.squeeze(image)
        image = np.stack((image,)*3, -1)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label

def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.augmentations.transforms.JpegCompression(p=0.5),
            A.augmentations.transforms.ImageCompression(p=0.5, compression_type=A.augmentations.transforms.ImageCompression.ImageCompressionType.WEBP),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        # self.model = torch.load('../models/resnext50_32x4d_ra-d733960d.pth')
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(
                   epoch+1, step+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   ))
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step+1, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

def train_loop(train_folds, valid_folds):

    LOGGER.info(f"========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_dataset = TrainDataset(train_folds, 
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, 
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, 
                              batch_size=CFG.batch_size, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.batch_size, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    def get_scheduler(optimizer):
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomResNext(CFG.model_name, pretrained=True)
    model.to(CFG.device)
    
    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = 0.
    best_loss = np.inf
    
    scores = []
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)
        
        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, CFG.device)
        valid_labels = valid_folds[CFG.target_col].values
        
        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds.argmax(1))

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Accuracy: {score}')
        
        scores.append(score)
        
        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_best.pth')
    
    check_point = torch.load(OUTPUT_DIR+f'{CFG.model_name}_best.pth')
    valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds, scores

def main(fold):
    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')
    
    def get_result2(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        matrix = get_confusion_matrix(labels, preds)
        print('TN', matrix[0,0])
        print('FP', matrix[0,1])
        print('FN', matrix[1,0])
        print('TP', matrix[1,1])
    
    # train 
    train_folds = folds.query(f'fold!={fold}').reset_index(drop=True)
    valid_folds = folds.query(f'fold=={fold}').reset_index(drop=False)
    oof_df, scores = train_loop(train_folds, valid_folds)
    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    get_result2(oof_df)
    # save result
    oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
    plt.plot([i for i in range(CFG.epochs)], scores)
    plt.title('valid score')
    plt.show()

if __name__ == '__main__':
    main(0)


# [Appendix 1] birdclef-mels-computer-public.ipynb
# 
# Generate  data: converting 7-sec audio to melspecs. The below script saves to "../output/02_appendix1/audio_images" but the pre-computed ones are in "../generated/kkiller-birdclef-mels-computer-d7-partx" for x in {1,2,3,4}.

# In[ ]:


import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
import joblib, json

from  sklearn.model_selection  import StratifiedKFold

PART_ID = 0 # The start index in the below list, by changing it you will compute mels on another subset
PART_INDEXES = [0,15718, 31436, 47154, 62874] # The train_set is splitted into 4 subsets

SR = 32_000
DURATION = 7 
SEED = 666

DATA_ROOT = Path("../input/birdclef-2021")
TRAIN_AUDIO_ROOT = Path("../input/birdclef-2021/train_short_audio")
TRAIN_AUDIO_IMAGES_SAVE_ROOT = Path("../output/02_appendix1/audio_images") # Where to save the mels images
TRAIN_AUDIO_IMAGES_SAVE_ROOT.mkdir(exist_ok=True, parents=True)

def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sr": sr, "duration": duration}

def make_df(n_splits=5, seed=SEED, nrows=None):
    
    df = pd.read_csv(DATA_ROOT/"train_metadata.csv", nrows=nrows)

    LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df["primary_label"].unique()))}
    
    df = df.iloc[PART_INDEXES[PART_ID]: PART_INDEXES[PART_ID+1]]

    df["label_id"] = df["primary_label"].map(LABEL_IDS)

    df["filepath"] = [str(TRAIN_AUDIO_ROOT/primary_label/filename) for primary_label,filename in zip(df.primary_label, df.filename) ]

    pool = joblib.Parallel(4)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.filepath]

    df = pd.concat([df, pd.DataFrame(pool(tqdm(tasks)))], axis=1, sort=False)
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = skf.split(np.arange(len(df)), y=df.label_id.values)
    df["fold"] = -1

    for fold, (train_set, val_set) in enumerate(splits):
        
        df.loc[df.index[val_set], "fold"] = fold

    return LABEL_IDS, df

LABEL_IDS, df = make_df(nrows=None)

df.to_csv("rich_train_metadata.csv", index=True)
with open("LABEL_IDS.json", "w") as f:
    json.dump(LABEL_IDS, f)

print(df.shape)
df.head()

class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):

        melspec = lb.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y

class AudioToImage:
    def __init__(self, sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast", resample=True):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin,
                                                 fmax=self.fmax)
        
    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio) 
        image = mono_to_color(melspec)
#         image = normalize(image, mean=None, std=None)
        return image

    def __call__(self, row, save=True):
#       max_audio_duration = 10*self.duration
#       init_audio_length = max_audio_duration*row.sr
        
#       start = 0 if row.duration <  max_audio_duration else np.random.randint(row.frames - init_audio_length)
    
      audio, orig_sr = sf.read(row.filepath, dtype="float32")

      if self.resample and orig_sr != self.sr:
        audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
        
      audios = [audio[i:i+self.audio_length] for i in range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
      audios[-1] = crop_or_pad(audios[-1] , length=self.audio_length)
      images = [self.audio_to_image(audio) for audio in audios]
      images = np.stack(images)
        
      if save:
        path = TRAIN_AUDIO_IMAGES_SAVE_ROOT/f"{row.primary_label}/{row.filename}.npy"
        path.parent.mkdir(exist_ok=True, parents=True)
        np.save(str(path), images)
      else:
        return  row.filename, images

def get_audios_as_images(df):
    pool = joblib.Parallel(2)
    
    converter = AudioToImage(step=int(DURATION*0.666*SR))
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]
    
    pool(tqdm(tasks))

get_audios_as_images(df)

row = df.loc[df.duration.idxmax()]
mels = np.load(str((TRAIN_AUDIO_IMAGES_SAVE_ROOT/row.primary_label/row.filename).as_posix() + ".npy"))
print(mels.shape)
lbd.specshow(mels[0])


# [Appendix 2] 
# 
# Use {birdclef-2021 training focals, 7-sec melspecs generated in Appendix 1, nocall detector models generated in Stage 1 (not made yet)} to output inference results on "../input/birdclef-2021/train_short_audio/" to "../output/03_appendix2/"

# In[ ]:


import torch

class CFG:
    debug = False
    print_freq=100
    num_workers=4
    model_name= 'resnext50_32x4d'
    dim=(128, 281)
    epochs=10
    batch_size=1
    seed=42
    target_size=2
    fold = 0 #choose from [0,1,2,3,4]
    pretrained = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
get_ipython().system('{sys.executable} -m pip install --quiet timm')

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import cv2
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations.pytorch.transforms import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

import warnings 
warnings.filterwarnings('ignore')

import glob

OUTPUT_DIR = '../output/03_appendix2/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

short = pd.read_csv('../input/birdclef-2021/train_metadata.csv')

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.filenames = df['filename'].values
        #self.labels = df['hasbird'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        filepath = glob.glob(f'../generated/kkiller-birdclef-mels-computer-d7-part*/audio_images/*/{file_name}.npy')[0]
        image = np.load(filepath)
        image = np.stack((image,)*3, -1)
        augmented_images = []
        if self.transform:
            for i in range(image.shape[0]):
                oneimage = image[i]
                augmented = self.transform(image=oneimage)
                oneimage = augmented['image']
                augmented_images.append(oneimage)
        #label = torch.tensor(self.labels[idx]).long()
        return np.stack(augmented_images, axis=0)#, label

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.augmentations.transforms.JpegCompression(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

import torch.nn as nn
import timm

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x

def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    count = 0
    for i, (images) in tk0:
        images = images[0]
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
        count += 1
        if count % 100 == 0:
            print(count)
    #probs = np.concatenate(probs)
    return np.asarray(probs)

if CFG.debug == True:
    short = short.sample(n=10)

# TODO put model from build_nocall_detector.ipynb (stage [1]) to be read here
MODEL_DIR = '../output/10_output_dir/clef-nocall-2class2-5fold/'
model = CustomResNext(CFG.model_name, pretrained=CFG.pretrained)
states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{CFG.fold}_best.pth'),]
test_dataset = TestDataset(short, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
predictions = inference(model, states, test_loader, CFG.device)

predictions = [i[:,1] for i in predictions]
predictions = [' '.join(map(str, j.tolist())) for j in predictions]
short['nocalldetection'] = predictions
short.to_csv(f'../output/03_appendix2/nocalldetection_for_shortaudio_fold{CFG.fold}.csv', index=False)

short.head()


# [2] build_melspectrogram_multilabel_classifier.ipynb:
# 
# Use: {7-sec melspecs generated in Appendix 1, data from 2020 competition, ff1010, nocall detector CSVs generated in Appendix 2 (not made myself yet, but avaiable in "../generated/train-short-audio-nocall-fold0to4/")} to output melspec multilabel classifier models to '../output/04_stage2'

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install -q scikit-learn==1.0.1 pysndfx SoundFile audiomentations pretrainedmodels efficientnet_pytorch resnest')

from pathlib import Path

import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from resnest.torch import resnest50

from matplotlib import pyplot as plt
import timm
import os, random, gc
import re, time, json
from  ast import literal_eval


from IPython.display import Audio
from sklearn.metrics import label_ranking_average_precision_score

from tqdm.notebook import tqdm
import joblib
import glob

from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import resnest.torch as resnest_torch

from sklearn.model_selection import StratifiedGroupKFold

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

NUM_CLASSES = 397
SR = 32_000
DURATION = 7

MAX_READ_SAMPLES = 10 # Each record will have 10 melspecs at most, you can increase this on Colab with High Memory Enabled

class Config:
    def __init__(self, debug:bool):
        self.debug = debug
        
        self.epochs = 1 if self.debug else 100 # 50

        self.max_distance = None # choose from [10, 20, None]
        if self.max_distance is not None:
            self.sites = ["SSW"] # choose multiples from ["COL", "COR", "SNE", "SSW"]
        else:
            self.sites = None
        self.max_duration = None # choose from [15, 30, 60, None]
        self.min_rating = None # choose from [3, 4, None], best: 3?
        self.max_spieces = None # choose from [100, 200, 300, None], best: 300?
        self.confidence_ub = 0.995 # Probability of birdsong occurrence, default: 0.995, choose from [0.5, 0.7, 0.9, 0.995]
        self.use_high_confidence_only = False # Whether to use only frames that are likely to be ringing (False performed better).
        self.use_mixup = True
        self.mixup_alpha = 0.5 # 5.0
        self.secondary_labels_weight = 0.6 #0.6 > 0.8 > 0.3 for better performance
        self.grouped_by_author = False
        self.folds = [0,]

        self.use_weight = False
        self.use_valid2020 = False
        self.use_ff1010 = False

        self.suffix = f"_sr{SR}_d{DURATION}"
        if self.max_spieces:
            self.suffix += f"_spices-{self.max_spieces}"
        if self.min_rating:
            self.suffix += f"_rating-{self.min_rating}"
        if self.use_high_confidence_only:
            self.suffix += f"_high-confidence-only"
        if self.use_mixup:
            self.suffix += f"_miixup-{self.mixup_alpha}"
        if self.secondary_labels_weight:
            self.suffix += f"_2ndlw-{self.secondary_labels_weight}"
        if self.use_weight:
            self.suffix += f"_weight"
        if self.use_valid2020:
            self.suffix += f"_valid2020"
        if self.use_ff1010:
            self.suffix += f"_ff1010"
        if self.grouped_by_author:
            self.suffix += f"_grouped-by-auther"

    def to_dict(self):
        return {
            "debug": self.debug,
            "epochs": self.epochs,
            "max_distance": self.max_distance,
            "sites": self.sites,
            "max_duration": self.max_duration,
            "min_rating": self.min_rating,
            "max_spieces": self.max_spieces,
            "confidence_ub": self.confidence_ub,
            "use_high_confidence_only": self.use_high_confidence_only,
            "use_mixup": self.use_mixup,
            "mixup_alpha": self.mixup_alpha,
            "secondary_labels_weight": self.secondary_labels_weight,
            "suffix": self.suffix,
            "grouped_by_author": self.grouped_by_author
        }

config = Config(debug=True)
from pprint import pprint
pprint(config.to_dict())

MODEL_NAMES = [
    # "resnext101_32x8d_wsl",
    # "resnest50",
    "resnest26d",
    # "tf_efficientnet_b0",
]

GENERATED = Path('../generated/')
MEL_PATHS = sorted(GENERATED.glob("kkiller-birdclef-mels-computer-d7-part?/rich_train_metadata.csv"))
TRAIN_LABEL_PATHS = sorted(GENERATED.glob("kkiller-birdclef-mels-computer-d7-part?/LABEL_IDS.json"))

MODEL_ROOT = Path("../output/04_stage2")

TRAIN_BATCH_SIZE = 64
TRAIN_NUM_WORKERS = 2

VAL_BATCH_SIZE = 64
VAL_NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

def get_df(mel_paths=MEL_PATHS, train_label_paths=TRAIN_LABEL_PATHS):
  df = None
  LABEL_IDS = {}
    
  for file_path in mel_paths:
    temp = pd.read_csv(str(file_path), index_col=0)
    temp["impath"] = temp.apply(lambda row: file_path.parent/"audio_images/{}/{}.npy".format(row.primary_label, row.filename), axis=1) 
    df = temp if df is None else df.append(temp)
    
  df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)

  for file_path in train_label_paths:
    with open(str(file_path)) as f:
      LABEL_IDS.update(json.load(f))

  return LABEL_IDS, df

from typing import List
def get_locations() -> List[dict]:
    return [{
        "site": "COL",
        "latitude": 5.57,
        "longitude": -75.85
    }, {
        "site": "COR",
        "latitude": 10.12,
        "longitude": -84.51
    }, {
        "site": "SNE",
        "latitude": 38.49,
        "longitude": -119.95
    }, {
        "site": "SSW",
        "latitude": 42.47,
        "longitude": -76.45
    }]

def is_in_site(row, sites, max_distance):
    for location in get_locations():
        if location["site"] in sites:
            x = (row["latitude"] - location["latitude"])
            y = (row["longitude"] - location["longitude"])
            r = (x**2 + y**2) ** 0.5
            if r < max_distance:
                return True
    return False

LABEL_IDS, df = get_df()

if config.grouped_by_author:
    kf = StratifiedGroupKFold(n_splits=5)
    x = df[["latitude", "longitude"]].values
    y = df["label_id"].values
    groups = df["author"].values
    df["fold"] = -1
    for kfold_index, (train_index, valid_index) in enumerate(kf.split(x, y, groups)):
        df.loc[valid_index, "fold"] = kfold_index

if config.debug:
    df = df.head(100)

print("before:%d" % len(df))
# Within a certain distance of the target area
if config.max_distance is not None:
    df = df[df.apply(lambda row: is_in_site(row, config.sites, config.max_distance), axis=1)]
# Number of Species
if config.max_spieces is not None:
    s = df["primary_label"].value_counts().head(config.max_spieces)
    df = df[df["primary_label"].isin(s.index)]
if config.min_rating is not None:
    df = df[df["rating"] >= config.min_rating]
if config.max_duration is not None:
    df = df[df["duration"] < config.max_duration]
df = df.reset_index(drop=True)
print("after:%d" % len(df))

print(df.shape)
df.head()

def get_model(name, num_classes=NUM_CLASSES):
    """
    Loads a pretrained model. 
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """

    if "resnest50" in name:
        if not os.path.exists("resnest50-528c19ca.pth"):
            get_ipython().system('wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth')
        pretrained_weights = torch.load('resnest50-528c19ca.pth')
        model = getattr(resnest_torch, name)(pretrained=False)
        model.load_state_dict(pretrained_weights)
    elif "resnest" in name:
        model = getattr(timm.models.resnest, name)(pretrained=True)
    elif name.startswith("resnext") or  name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("tf_efficientnet"):
        model = getattr(timm.models.efficientnet, name)(pretrained=True)
    elif "efficientnet-b" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)

    return model

import glob
nocall_paths = glob.glob("../generated/train-short-audio-nocall-fold0to4/train_short_audio_nocall_fold0to4/*.csv")
probs_list = []
for nocall_path in nocall_paths:
    nocall_df = pd.read_csv(nocall_path)
    probs = nocall_df["nocalldetection"].apply(
        lambda _: list(
            map(
                float,
                _.split()
            )
        )
    )
    probs_list.append(probs)
probs = []
for di in range(len(nocall_df)):
    one_row = []
    for ni in range(len(nocall_paths)):
        one_row.append(probs_list[ni][di])
    probs.append(np.mean(one_row,axis=0).tolist())

audio_prob_store = dict(zip(nocall_df["filename"].tolist(), probs))

# TODO this is set to false; investigate where to get this data from?
if config.use_valid2020:
    clef_2020_df = pd.read_csv("../input/birdclef2020-validation-audio-and-ground-truth-d5/rich_metadata.csv", index_col=0)
    clef_2020_df["fold"] = clef_2020_df["file_fold"]%5
    clef_2020_df["impath"] = "../input/birdclef2020-validation-audio-and-ground-truth-d5/" + clef_2020_df["primary_label"] + "/" + clef_2020_df["filename"] + ".npy"
    clef_2020_df["label_id"] = -1
    clef_2020_df = clef_2020_df[clef_2020_df["primary_label"]=="nocall"]
    clef_2020_df["secondary_labels"] = [[] for i in range(len(clef_2020_df))]

    # Update prob with nocall detector
    probs = [[0] for i in range(len(clef_2020_df))]
    prob_dict = dict(zip(clef_2020_df["filename"].tolist(), probs))
    audio_prob_store.update(prob_dict)

    df = pd.concat([clef_2020_df, df]).reset_index(drop=True)

if config.use_ff1010:
    ff1010_df = pd.read_csv("../input/ff1010bird-duration7-1/rich_metadata.csv", index_col=0)
    ff1010_df["impath"] = "../input/ff1010bird-duration7-1/" + ff1010_df["primary_label"] + "/" + ff1010_df["filename"] + ".npy"
    ff1010_df = ff1010_df[ff1010_df["primary_label"]=="nocall"]
    ff1010_df["label_id"] = -1
    ff1010_df["fold"] = ff1010_df.index % 5
    ff1010_df["secondary_labels"] = [[] for i in range(len(ff1010_df))]

    # Update prob with nocall detector
    probs = [[0] for i in range(len(ff1010_df))]
    prob_dict = dict(zip(ff1010_df["filename"].tolist(), probs))
    audio_prob_store.update(prob_dict)

    df = pd.concat([ff1010_df, df]).reset_index(drop=True)

def load_data(df):
    def load_row(row):
        # impath = TRAIN_IMAGES_ROOT/f"{row.primary_label}/{row.filename}.npy"
        return row.filename, np.load(str(row.impath))[:MAX_READ_SAMPLES]
    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row)
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res

# We cache the train set to reduce training time

audio_image_store = load_data(df)
len(audio_image_store)

# print("shape:", next(iter(audio_image_store.values())).shape)
# lbd.specshow(next(iter(audio_image_store.values()))[0])

for k, v in LABEL_IDS.items():
    print(k, v)
    break

image_w = 281

def pad_image(image, image_w=image_w):
    h = image.shape[0]
    w = image.shape[1]
    if w < image_w:
        start = np.random.choice((image_w-w))
        ret = np.zeros((h, image_w))
        ret[:, start:start+w] = image
        return ret
    return image


class BirdClefDataset(Dataset):

    def __init__(
        self,
        audio_image_store,
        audio_prob_store,
        meta,
        sr=SR,
        is_train=True,
        num_classes=NUM_CLASSES,
        duration=DURATION,
    ):        
        self.audio_image_store = audio_image_store
        self.audio_prob_store = audio_prob_store
        self.meta = meta.copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.eps = 0.0025
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.meta)
    
    def mixup_data(self, image, noize, alpha=0.5):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        lam /= 2 # leave it at half maximum.
        mixed_x = (1 - lam) * image + lam * noize
        return mixed_x

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        images = self.audio_image_store[row.filename]
        probs = self.audio_prob_store[row.filename]

        i = np.random.choice(len(images))
        image = images[i]

        if image.shape[1] < self.audio_length:
            image = pad_image(image, image_w)

        
        image = self.normalize(image)
        prob = probs[i]
        t = np.zeros(self.num_classes, dtype=np.float32) + self.eps # Label smoothing
        t[row.label_id] = max(min(prob, config.confidence_ub), self.eps) # clipping
        for secondary_label in row.secondary_labels:
            # Set a lower value than the primary label
            if secondary_label in LABEL_IDS:
                t[LABEL_IDS[secondary_label]] = max(self.eps, prob * 0.6)

        
        return image, t

ds = BirdClefDataset(audio_image_store, audio_prob_store, meta=df, sr=SR, duration=DURATION, is_train=True)
# len(df)

# x, y = ds[np.random.choice(len(ds))]
# # x, y = ds[0]
# print(x.shape, y.shape, np.where(y >= 0.5))
# lbd.specshow(x[0])

# TODO set use_cuda=True on cluster
def mixup_data(x, y, alpha=0.5, use_cuda=False):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_y = lam * y + (1 - lam) * y[index]
    #mixed_y = torch.maximum(y, y[index])
    return mixed_x, mixed_y, lam

def one_step( xb, yb, net, criterion, optimizer, scheduler=None):
  if config.use_mixup:
      xb, yb, lam = mixup_data(xb,yb, alpha=config.mixup_alpha)
  xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
  optimizer.zero_grad()
  o = net(xb)
  loss = criterion(o, yb)
  loss.backward()
  optimizer.step()
  
  with torch.no_grad():
      l = loss.item()

      o = o.sigmoid()
      yb = (yb > 0.5 )*1.0
      lrap = label_ranking_average_precision_score(yb.cpu().numpy(), o.cpu().numpy())

      o = (o > 0.5)*1.0

      prec = (o*yb).sum()/(1e-6 + o.sum())
      rec = (o*yb).sum()/(1e-6 + yb.sum())
      f1 = 2*prec*rec/(1e-6+prec+rec)

  if  scheduler is not None:
    scheduler.step()

  return l, lrap, f1.item(), rec.item(), prec.item()

@torch.no_grad()
def evaluate(net, criterion, val_laoder):
    net.eval()

    os, y = [], []
    val_laoder = tqdm(val_laoder, leave = False, total=len(val_laoder))

    for icount, (xb, yb) in  enumerate(val_laoder):

        y.append(yb.to(DEVICE))

        xb = xb.to(DEVICE)
        o = net(xb)

        os.append(o)

    y = torch.cat(y)
    o = torch.cat(os)

    l = criterion(o, y).item()
    
    o = o.sigmoid()
    y = (y > 0.5)*1.0

    lrap = label_ranking_average_precision_score(y.cpu().numpy(), o.cpu().numpy())

    o = (o > 0.5)*1.0

    prec = ((o*y).sum()/(1e-6 + o.sum())).item()
    rec = ((o*y).sum()/(1e-6 + y.sum())).item()
    f1 = 2*prec*rec/(1e-6+prec+rec)

    return l, lrap, f1, rec, prec

def one_epoch(net, criterion, optimizer, scheduler, train_laoder, val_laoder):
  net.train()
  l, lrap, prec, rec, f1, icount = 0.,0.,0.,0., 0., 0
  train_laoder = tqdm(train_laoder, leave = False)
  epoch_bar = train_laoder
  
  for (xb, yb) in  epoch_bar:
      # epoch_bar.set_description("----|----|----|----|---->")
      _l, _lrap, _f1, _rec, _prec = one_step(xb, yb, net, criterion, optimizer)
      l += _l
      lrap += _lrap
      f1 += _f1
      rec += _rec
      prec += _prec

      icount += 1
        
      if hasattr(epoch_bar, "set_postfix") and not icount%10:
          epoch_bar.set_postfix(
            loss="{:.6f}".format(l/icount),
            lrap="{:.3f}".format(lrap/icount),
            prec="{:.3f}".format(prec/icount),
            rec="{:.3f}".format(rec/icount),
            f1="{:.3f}".format(f1/icount),
          )
  
  scheduler.step()

  l /= icount
  lrap /= icount
  f1 /= icount
  rec /= icount
  prec /= icount
  
  l_val, lrap_val, f1_val, rec_val, prec_val = evaluate(net, criterion, val_laoder)
  
  return (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val)

class AutoSave:
  def __init__(self, top_k=50, metric="f1", mode="min", root=None, name="ckpt"):
    self.top_k = top_k
    self.logs = []
    self.metric = metric
    self.mode = mode
    self.root = Path(root or MODEL_ROOT)
    assert self.root.exists()
    self.name = name

    self.top_models = []
    self.top_metrics = []

  def log(self, model, metrics):
    metric = metrics[self.metric]
    rank = self.rank(metric)

    self.top_metrics.insert(rank+1, metric)
    if len(self.top_metrics) > self.top_k:
      self.top_metrics.pop(0)

    self.logs.append(metrics)
    self.save(model, metric, rank, metrics["epoch"])


  def save(self, model, metric, rank, epoch):
    t = time.strftime("%Y%m%d%H%M%S")
    name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(self.name, epoch, self.metric, metric, t)
    name = re.sub(r"[^\w_-]", "", name) + ".pth"
    path = self.root.joinpath(name)

    old_model = None
    self.top_models.insert(rank+1, name)
    if len(self.top_models) > self.top_k:
      old_model = self.root.joinpath(self.top_models[0])
      self.top_models.pop(0)      

    torch.save(model.state_dict(), path.as_posix())

    if old_model is not None:
      old_model.unlink()

    self.to_json()


  def rank(self, val):
    r = -1
    for top_val in self.top_metrics:
      if val <= top_val:
        return r
      r += 1

    return r
  
  def to_json(self):
    # t = time.strftime("%Y%m%d%H%M%S")
    name = "{}_logs".format(self.name)
    name = re.sub(r"[^\w_-]", "", name) + ".json"
    path = self.root.joinpath(name)

    with path.open("w") as f:
      json.dump(self.logs, f, indent=2)

def one_fold(model_name, fold, train_set, val_set, epochs=20, save=True, save_root=None):

  save_root = Path(save_root) or MODEL_ROOT

  saver = AutoSave(root=save_root, name=f"birdclef_{model_name}_fold{fold}", metric="f1_val")

  net = get_model(model_name).to(DEVICE)

  #criterion = nn.BCEWithLogitsLoss()
  weight = None
  if config.use_weight:
      label_inv = (1/df["label_id"].value_counts().sort_index()).values
      label_inv_mean = label_inv.mean()
      weight = label_inv*(1/label_inv_mean)  # Inverse proportion such that the mean is 1
      weight = torch.tensor(weight).to(DEVICE)
  criterion = nn.BCEWithLogitsLoss(weight=weight)
  
  lr =  8e-4
  optimizer = optim.Adam(net.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=epochs)

  train_data = BirdClefDataset(
      audio_image_store,
      audio_prob_store,
      meta=df.iloc[train_set].reset_index(drop=True),
      sr=SR,
      duration=DURATION,
      is_train=True
    )
  train_laoder = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, pin_memory=True)

  val_data = BirdClefDataset(
      audio_image_store,
      audio_prob_store,
      meta=df.iloc[val_set].reset_index(drop=True),
      sr=SR,
      duration=DURATION,
      is_train=False)
  val_laoder = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False)

  epochs_bar = tqdm(list(range(epochs)), leave=False)
  for epoch  in epochs_bar:
    epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
    net.train()

    (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val) = one_epoch(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_laoder=train_laoder,
        val_laoder=val_laoder,
      )

    epochs_bar.set_postfix(
        loss="({:.6f}, {:.6f})".format(l, l_val),
        prec="({:.3f}, {:.3f})".format(prec, prec_val),
        rec="({:.3f}, {:.3f})".format(rec, rec_val),
        f1="({:.3f}, {:.3f})".format(f1, f1_val),
        lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
    )

    print(
        "[{epoch:02d}] loss: {loss} lrap: {lrap} f1: {f1} rec: {rec} prec: {prec}".format(
            epoch=epoch,
            loss="({:.6f}, {:.6f})".format(l, l_val),
            prec="({:.3f}, {:.3f})".format(prec, prec_val),
            rec="({:.3f}, {:.3f})".format(rec, rec_val),
            f1="({:.3f}, {:.3f})".format(f1, f1_val),
            lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
        )
    )

    if save:
      metrics = {
          "loss": l, "lrap": lrap, "f1": f1, "rec": rec, "prec": prec,
          "loss_val": l_val, "lrap_val": lrap_val, "f1_val": f1_val, "rec_val": rec_val, "prec_val": prec_val,
          "epoch": epoch,
      }

      saver.log(net, metrics)

def train(model_name, epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix="", folds=None):
  gc.collect()
  torch.cuda.empty_cache()

  save_root = save_root or MODEL_ROOT/f"{model_name}{suffix}"
  save_root.mkdir(exist_ok=True, parents=True)
  
  fold_bar = tqdm(df.reset_index().groupby("fold").index.apply(list).items(), total=df.fold.max()+1)
  
  for fold, val_set in fold_bar:
      if folds and not fold in folds:
        continue
      
      print(f"\n############################### [FOLD {fold}]")
      fold_bar.set_description(f"[FOLD {fold}]")
      train_set = np.setdiff1d(df.index, val_set)
        
      one_fold(model_name, fold=fold, train_set=train_set , val_set=val_set , epochs=epochs, save=save, save_root=save_root)
    
      gc.collect()
      torch.cuda.empty_cache()

for model_name in MODEL_NAMES:
    print("\n\n###########################################", model_name.upper())
    train(model_name, epochs=config.epochs, suffix=config.suffix, folds=config.folds)


# [Appendix 3] calculate_397dimprobs_for_train_short_audio.ipynb:
# 

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install -q pysndfx SoundFile audiomentations pretrainedmodels efficientnet_pytorch resnest')
get_ipython().system('{sys.executable} -m pip install timm')

import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path

import torch
from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader

from resnest.torch import resnest50

from matplotlib import pyplot as plt

import os, random, gc
import re, time, json
from  ast import literal_eval

from IPython.display import Audio
from sklearn.metrics import label_ranking_average_precision_score

from tqdm.notebook import tqdm
import joblib

import timm
from sklearn.model_selection import StratifiedGroupKFold

from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import resnest.torch as resnest_torch

if not os.path.exists("../generated/resnest50-528c19ca.pth"):
    get_ipython().system('wget  "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth"')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

NUM_CLASSES = 397
SR = 32_000
DURATION = 7

MAX_READ_SAMPLES = 15 # Each record will have 10 melspecs at most, you can increase this on Colab with High Memory Enabled

class Config:
    def __init__(self, debug:bool):
        self.debug = debug
        
        self.epochs = 1 if self.debug else 50

        self.max_distance = None # choose from [10, 20, None]
        if self.max_distance is not None:
            self.sites = ["SSW"] # choose multiples from ["COL", "COR", "SNE", "SSW"]
        else:
            self.sites = None
        self.max_duration = None # choose from [15, 30, 60, None]
        self.min_rating = None # choose from [3, 4, None], best: 3?
        self.max_spieces = None # choose from [100, 200, 300, None], best: 300?
        self.confidence_ub = 0.995 # Probability of birdsong occurrence, default: 0.995, choose from [0.5, 0.7, 0.9, 0.995]
        self.use_high_confidence_only = False # Whether to use only frames that are likely to be ringing (False performed better).
        self.use_mixup = True
        self.mixup_alpha = 5.0 # 0.5
        self.grouped_by_author = True
        # self.folds = [4]

        self.suffix = f"sr{SR}_d{DURATION}"
        if self.max_spieces:
            self.suffix += f"_spices-{self.max_spieces}"
        if self.min_rating:
            self.suffix += f"_rating-{self.min_rating}"
        if self.use_high_confidence_only:
            self.suffix += f"_high-confidence-only"
        if self.use_mixup:
            self.suffix += f"_miixup-{self.mixup_alpha}"
        if self.grouped_by_author:
            self.suffix += f"_grouped-by-auther"

    def to_dict(self):
        return {
            "debug": self.debug,
            "epochs": self.epochs,
            "max_distance": self.max_distance,
            "sites": self.sites,
            "max_duration": self.max_duration,
            "min_rating": self.min_rating,
            "max_spieces": self.max_spieces,
            "confidence_ub": self.confidence_ub,
            "use_high_confidence_only": self.use_high_confidence_only,
            "use_mixup": self.use_mixup,
            "mixup_alpha": self.mixup_alpha,
            "suffix": self.suffix,
            "grouped_by_author": self.grouped_by_author
        }

config = Config(debug=False)
from pprint import pprint
pprint(config.to_dict())

MODEL_NAMES = [
    # "resnext101_32x8d_wsl",
    # 'efficientnet_b0',
    "resnest50",
    # "densenet121",
] 

MEL_PATHS = sorted(Path("../generated").glob("kkiller-birdclef-mels-computer-d7-part?/rich_train_metadata.csv"))
TRAIN_LABEL_PATHS = sorted(Path("../generated").glob("kkiller-birdclef-mels-computer-d7-part?/LABEL_IDS.json"))

TRAIN_BATCH_SIZE = 50 # 16
TRAIN_NUM_WORKERS = 2

VAL_BATCH_SIZE = 50 # 16 # 128
VAL_NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

checkpoint_paths = [
    # model1
    Path("../generated/clefmodel/birdclef_resnest50_fold0_epoch_27_f1_val_05179_20210520120053.pth"),
]

def get_df(mel_paths=MEL_PATHS, train_label_paths=TRAIN_LABEL_PATHS):
  df = None
  LABEL_IDS = {}
    
  for file_path in mel_paths:
    temp = pd.read_csv(str(file_path), index_col=0)
    temp["impath"] = temp.apply(lambda row: file_path.parent/"audio_images/{}/{}.npy".format(row.primary_label, row.filename), axis=1) 
    df = temp if df is None else df.append(temp)
    
  df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)

  for file_path in train_label_paths:
    with open(str(file_path)) as f:
      LABEL_IDS.update(json.load(f))

  return LABEL_IDS, df

from typing import List
def get_locations() -> List[dict]:
    return [{
        "site": "COL",
        "latitude": 5.57,
        "longitude": -75.85
    }, {
        "site": "COR",
        "latitude": 10.12,
        "longitude": -84.51
    }, {
        "site": "SNE",
        "latitude": 38.49,
        "longitude": -119.95
    }, {
        "site": "SSW",
        "latitude": 42.47,
        "longitude": -76.45
    }]

def is_in_site(row, sites, max_distance):
    for location in get_locations():
        if location["site"] in sites:
            x = (row["latitude"] - location["latitude"])
            y = (row["longitude"] - location["longitude"])
            r = (x**2 + y**2) ** 0.5
            if r < max_distance:
                return True
    return False

LABEL_IDS, df = get_df()

if config.grouped_by_author:
    kf = StratifiedGroupKFold(n_splits=5)
    x = df[["latitude", "longitude"]].values
    y = df["label_id"].values
    groups = df["author"].values
    df["fold"] = -1
    for kfold_index, (train_index, valid_index) in enumerate(kf.split(x, y, groups)):
        df.loc[valid_index, "fold"] = kfold_index

if config.debug:
    df = df.head(100)

print("before:%d" % len(df))
# Within a certain distance of the target area
if config.max_distance is not None:
    df = df[df.apply(lambda row: is_in_site(row, config.sites, config.max_distance), axis=1)]
# Number of Species
if config.max_spieces is not None:
    s = df["primary_label"].value_counts().head(config.max_spieces)
    df = df[df["primary_label"].isin(s.index)]
# Rating is above a certain value
if config.min_rating is not None:
    df = df[df["rating"] >= config.min_rating]
# Within a certain amount of recording time
if config.max_duration is not None:
    df = df[df["duration"] < config.max_duration]
df = df.reset_index(drop=True)
print("after:%d" % len(df))

print(df.shape)
df.head()

def get_model(name, num_classes=NUM_CLASSES):
    """
    Loads a pretrained model. 
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if "resnest" in name:
        if not os.path.exists("../generated/resnest50-528c19ca.pth"):
            get_ipython().system('wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth')
    
        pretrained_weights = torch.load('../generated/resnest50-528c19ca.pth')
        model = getattr(resnest_torch, name)(pretrained=False)
        model.load_state_dict(pretrained_weights)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("resnext") or  name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif name.startswith("efficientnet_b"):
        model = getattr(timm.models.efficientnet, name)(pretrained=True)
    elif name.startswith("densenet"):
        model = getattr(timm.models.densenet, name)(pretrained=True)
    elif "efficientnet-b" in name:
        model = EfficientNet.from_pretrained(name)
    else:
        model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)

    return model

class BirdClefDataset(Dataset):
    def __init__(
        self,
        meta,
        sr=SR,
        is_train=True,
        num_classes=NUM_CLASSES,
        duration=DURATION
    ):
        self.meta = meta.copy().reset_index(drop=True)
        records = []
        for idx, row in tqdm(self.meta.iterrows(), total=len(self.meta)):
            images = np.load(str(row["impath"]))
            for i, image in enumerate(images):
                seconds = i * duration
                records.append({
                    "filename": row["filename"],
                    "impath": row["impath"],
                    "seconds": seconds,
                    "index": i
                })
        self.records = records
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.eps = 0.0025
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        image = np.load(str(row["impath"]))[row["index"]]
        image = self.normalize(image)
        return image, row["filename"], row["seconds"]

nocall_df = pd.read_csv("../generated/train-short-audio-nocall-fold0to4/train_short_audio_nocall_fold0to4/nocalldetection_for_shortaudio_fold0.csv")

ds = BirdClefDataset(meta=df, sr=SR, duration=DURATION, is_train=True)
len(df)

def add_tail(model, num_classes):
    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)
    return model

def load_net(checkpoint_path, num_classes=NUM_CLASSES):
    if "resnest50" in checkpoint_path:
        net = resnest50(pretrained=False)
    elif "resnest26d" in checkpoint_path:
        net = timm.models.resnest26d(pretrained=False)
    elif "resnext101_32x8d_wsl" in checkpoint_path:
        net = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    elif "efficientnet_b0" in checkpoint_path:
        net = getattr(timm.models.efficientnet, "efficientnet_b0")(pretrained=False)
    elif "densenet121" in checkpoint_path:
        net = timm.models.densenet121(pretrained=False)
    else:
        raise ValueError("Unexpected checkpont name: %s" % checkpoint_path)
    net = add_tail(net, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net

@torch.no_grad()
def predict(net, criterion, val_laoder):
    net.eval()
    records = []
    val_laoder = tqdm(val_laoder, leave = False, total=len(val_laoder))
    for icount, (xb, filename, seconds) in enumerate(val_laoder):
        xb = xb.to(DEVICE)
        prob = net(xb)
        prob = torch.sigmoid(prob)
        records.append({
            "prob": prob,
            "filename": filename,
            "seconds": seconds
        })
    return records

def one_fold(checkpoint_path, fold, train_set, val_set, epochs=20, save=True, save_root=None):
    net = load_net(checkpoint_path)
    criterion = nn.BCEWithLogitsLoss()
    val_data = BirdClefDataset(
        meta=df.iloc[val_set].reset_index(drop=True),
        sr=SR,
        duration=DURATION,
        is_train=False
    )
    val_laoder = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False)
    y_preda = predict(net, criterion, val_laoder)
    return y_preda

def predict_for_oof(checkpoint_path, epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix="", folds=None):
  gc.collect()
  torch.cuda.empty_cache()

  fold_bar = tqdm(df.reset_index().groupby("fold").index.apply(list).items(), total=df.fold.max()+1)
  
  for fold, val_set in fold_bar:
      if folds and not fold in folds:
        continue
      
      print(f"\n############################### [FOLD {fold}]")
      fold_bar.set_description(f"[FOLD {fold}]")
      train_set = np.setdiff1d(df.index, val_set)
      records = one_fold(checkpoint_path, fold=fold, train_set=train_set , val_set=val_set , epochs=epochs, save=save, save_root=save_root)
      gc.collect()
      torch.cuda.empty_cache()
      return records

def to_call_prob(row):
    i = row["seconds"] // DURATION
    call_prob = float(row["nocalldetection"].split()[i])
    return call_prob

def to_birds(row):
    if row["call_prob"] < 0.5:
        return "nocall"
    res = [row["primary_label"]] + eval(row["secondary_labels"])
    return " ".join(res)

INV_LABEL_IDS = {v:k for k, v in LABEL_IDS.items()}
columns = [INV_LABEL_IDS[i] for i in range(len(LABEL_IDS))]
metadata_df = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
nocall_df = pd.read_csv("../input/train-short-audio-nocall-fold0to4/train_short_audio_nocall_fold0to4/nocalldetection_for_shortaudio_fold0.csv")

import glob
filename_to_nocalldetection = {}
filepath_list = list(glob.glob("../generated/train-short-audio-nocall-fold0to4/train_short_audio_nocall_fold0to4/*.csv"))
for filepath in filepath_list:
    nocall_df = pd.read_csv(filepath)
    probs = nocall_df["nocalldetection"].apply(
        lambda _: list(
            map(float, _.split())
        )
    ).tolist()
    for k, v in zip(nocall_df["filename"].tolist(), probs):
        if not k in filename_to_nocalldetection:
            filename_to_nocalldetection[k] = v
        else:
            w = filename_to_nocalldetection[k]
            for i in range( len(w)):
                w[i] += v[i]
            filename_to_nocalldetection[k] = w

for k, v in filename_to_nocalldetection.items():
    for i in range(len(v)):
        filename_to_nocalldetection[k][i] /= len(filepath_list)

for k, v in filename_to_nocalldetection.items():
    filename_to_nocalldetection[k] = " ".join(map(str, v))

nocall_df = pd.DataFrame(filename_to_nocalldetection.items(), columns=["filename", "nocalldetection"])

nocall_df.head()

filepath_list = []
for checkpoint_path in checkpoint_paths:
    print("\n\n###########################################", checkpoint_path)
    # Find out which fold it is from the name of the model file.
    fold = -1
    for i in range(5):
        if f"fold{i}" in checkpoint_path.stem:
            fold = i
            break
    print("target validation fold is %d" % fold)
    if fold == -1:
        raise ValueError("Unexpected fold value")
    # Run on the fold that is the target of oof.
    records_list = predict_for_oof(checkpoint_path.as_posix(), epochs=config.epochs, suffix=config.suffix, folds=[fold])
    dfs = []
    for records in records_list:
        prob = records["prob"].to("cpu").numpy()
        _df = pd.DataFrame(prob)
        _df.columns = columns
        _df["seconds"] = records["seconds"].to("cpu").numpy().tolist()
        _df["filename"] = list(records["filename"])
        dfs.append(_df)
    oof_df = pd.concat(dfs)
    oof_df = pd.merge(oof_df, metadata_df, how="left", on=["filename"])
    oof_df = pd.merge(oof_df, nocall_df[["filename", "nocalldetection"]], how="left", on=["filename"])
    oof_df["call_prob"] = oof_df.apply(to_call_prob, axis=1)
    oof_df["birds"] = oof_df.apply(to_birds, axis=1)
    filepath = "%s.csv" % checkpoint_path.stem
    print(f"Save to {filepath}")
    oof_df.drop(
        columns=[
            'scientific_name',
            'common_name',
            'license',
            'time',
            'url',
            'nocalldetection',
        ]
    ).to_csv(filepath, index=False)
    filepath_list.append(filepath)

oof_df.drop(
    columns=[
        'scientific_name',
        'common_name',
        'license',
        'time',
        'url',
        'nocalldetection',
    ]
)


# 60_input_fresh.ipynb:
# 
# Use {melspec multilabel classifier models generated in stage 2 (not generated by me yet but available in "../generated/birdclef-groupby-author-05221040-728258/"), "clefmodel", multilabel classifier models generated in Appendix 3 (not made by me yet but available at "../generated/metadata-probability-v0525-2100/") "../generated/resnest50-fast-package/" library) to yield "../output/submission.csv"

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install -q --use-feature=in-tree-build "../generated/resnest50-fast-package/resnest-0.0.6b20200701/resnest"')

import warnings
warnings.filterwarnings('ignore')

LABELS_PROBABILITY_DIR = "../output/06_stage3/labels_probability/"
LGBM_PKL_DIR = "../output/06_stage3/lgbm_pkl/"

import os
if not os.path.exists(LABELS_PROBABILITY_DIR):
    os.mkdir(LABELS_PROBABILITY_DIR)
if not os.path.exists(LGBM_PKL_DIR):
    os.mkdir(LGBM_PKL_DIR)

import numpy as np
import pandas as pd
from pathlib import Path

import re
import time

import pickle
from typing import List
from tqdm.notebook import tqdm

# sound
import librosa as lb
import soundfile as sf

# pytorch
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50

import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
import pickle
from catboost import CatBoostClassifier
from catboost import Pool
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import random

BIRD_LIST = sorted(os.listdir('../input/birdclef-2021/train_short_audio'))
BIRD2IDX = {bird:idx for idx, bird in enumerate(BIRD_LIST)}
BIRD2IDX['nocall'] = -1
IDX2BIRD = {idx:bird for bird, idx in BIRD2IDX.items()}

# 10 frame version
filepath_list = [
     "../generated/metadata-probability-v0525-2100/birdclef_resnest50_fold1_epoch_34_f1_val_04757_20210524185455.csv"
]
prob_df = pd.concat([pd.read_csv(_) for _ in filepath_list])

class TrainingConfig:
    def __init__(self):
        self.nocall_threshold:float=0.5
        self.num_kfolds:int = 5
        self.num_spieces:int = 397
        self.num_candidates:int = 5
        self.max_distance:int = 15 # 20
        self.sampling_strategy:float = None # 1.0
        self.random_state:int=777
        self.num_prob:int = 6
        self.use_to_birds=True
        self.weights_filepath_dict = {
            'lgbm':[(LGBM_PKL_DIR + f"lgbm_{kfold_index}.pkl") for kfold_index in range(self.num_kfolds)],
        }
        
training_config = TrainingConfig()

class Config:
    def __init__(self):
        self.num_kfolds:int = training_config.num_kfolds
        self.num_spieces:int = training_config.num_spieces
        self.num_candidates:int = training_config.num_candidates
        self.max_distance:int = training_config.max_distance
        self.nocall_threshold:float = training_config.nocall_threshold
        self.num_prob:int = training_config.num_prob
        # check F1 score without 3rd stage(table competition) 
        self.check_baseline:bool = True
        # List of file paths of the models which are used when determining if the bird is acceptable.
        self.weights_filepath_dict = training_config.weights_filepath_dict
        # Weights for the models to predict the probability of each bird singing for each frame.
        self.checkpoint_paths = [ 
            Path("../generated/clefmodel/birdclef_resnest50_fold0_epoch_27_f1_val_05179_20210520120053.pth"), # id36
            Path("../generated/clefmodel/birdclef_resnest50_fold0_epoch_13_f1_val_03502_20210522050604.pth"), # id51
            Path("../generated/birdclef-groupby-author-05221040-728258/birdclef_resnest50_fold0_epoch_33_f1_val_03859_20210524151554.pth"), # id58
            Path("../generated/birdclef-groupby-author-05221040-728258/birdclef_resnest50_fold1_epoch_34_f1_val_04757_20210524185455.pth"), # id59
            Path("../generated/birdclef-groupby-author-05221040-728258/birdclef_resnest50_fold2_epoch_34_f1_val_05027_20210524223209.pth"), # id60
            Path("../generated/birdclef-groupby-author-05221040-728258/birdclef_resnest50_fold3_epoch_20_f1_val_04299_20210525010703.pth"), # id61
            Path("../generated/birdclef-groupby-author-05221040-728258/birdclef_resnest50_fold4_epoch_34_f1_val_05140_20210525074929.pth"), # id62
            Path("../generated/clefmodel/resnest50_sr32000_d7_miixup-5.0_2ndlw-0.6_grouped-by-auther/birdclef_resnest50_fold0_epoch_78_f1_val_03658_20210528221629.pth"), # id97
            Path("../generated/clefmodel/resnest50_sr32000_d7_miixup-5.0_2ndlw-0.6_grouped-by-auther/birdclef_resnest50_fold0_epoch_84_f1_val_03689_20210528225810.pth"), # id97
            Path("../generated/clefmodel/resnest50_sr32000_d7_miixup-5.0_2ndlw-0.6_grouped-by-auther/birdclef_resnest50_fold1_epoch_27_f1_val_03942_20210529062427.pth"), # id98
        ]
        # call probability of each bird for each sample used for candidate extraction (cache)
        self.pred_filepath_list = [
            self.get_prob_filepath_from_checkpoint(path) for path in self.checkpoint_paths
        ]

    def get_prob_filepath_from_checkpoint(self, checkpoint_path:Path) -> str:
        return LABELS_PROBABILITY_DIR + "train_soundscape_labels_probabilitiy_" + checkpoint_path.stem + ".csv"

config = Config()

def get_locations():
    return [{
        "site": "COL",
        "latitude": 5.57,
        "longitude": -75.85
    }, {
        "site": "COR",
        "latitude": 10.12,
        "longitude": -84.51
    }, {
        "site": "SNE",
        "latitude": 38.49,
        "longitude": -119.95
    }, {
        "site": "SSW",
        "latitude": 42.47,
        "longitude": -76.45
    }]


def to_site(row, max_distance:int):
    best = max_distance
    answer = "Other"
    for location in get_locations():
        x = (row["latitude"] - location["latitude"])
        y = (row["longitude"] - location["longitude"])
        dist = (x**2 + y**2) ** 0.5
        if dist < best:
            best = dist
            answer = location["site"]
    return answer


def to_latitude(site:str) -> str:
    for location in get_locations():
        if site == location["site"]:
            return location["latitude"]
    return -10000


def to_longitude(site:str) -> str:
    for location in get_locations():
        if site == location["site"]:
            return location["longitude"]
    return -10000


def to_birds(row, th:float) -> str:
    if row["call_prob"] < th:
        return "nocall"
    res = [row["primary_label"]] + eval(row["secondary_labels"])
    return " ".join(res)

def make_candidates(
    prob_df:pd.DataFrame,
    num_spieces:int,
    num_candidates:int,
    max_distance:int,
    num_prob:int=6, # number of frames to be allocated for front and rear (if 3, then 3 for front, 3 for rear)
    nocall_threshold:float=0.5,
):
    if "author" in prob_df.columns: # meta data (train_short_audio)
        prob_df["birds"] = prob_df.apply(
            lambda row: to_birds(row, th=nocall_threshold),
            axis=1
        )
        print("Candidate nocall ratio: %.4f" % (prob_df["birds"] == "nocall").mean())
        prob_df["audio_id"] = prob_df["filename"].apply(
            lambda _: int(_.replace("XC", "").replace(".ogg", ""))
        )
        prob_df["row_id"] = prob_df.apply(
            lambda row: "%s_%s" % (row["audio_id"], row["seconds"]),
            axis=1
        )
        prob_df["year"] = prob_df["date"].apply(lambda _: int(_.split("-")[0]))
        prob_df["month"] = prob_df["date"].apply(lambda _: int(_.split("-")[1]))
        prob_df["site"] = prob_df.apply(
            lambda row: to_site(row, max_distance),
            axis=1
        )
    else:
        prob_df["year"] = prob_df["date"].apply(lambda _: int(str(_)[:4]))
        prob_df["month"] = prob_df["date"].apply(lambda _: int(str(_)[4:6]))
        prob_df["latitude"] = prob_df["site"].apply(to_latitude)
        prob_df["longitude"] = prob_df["site"].apply(to_longitude)
        
    sum_prob_list = prob_df[BIRD_LIST].sum(axis=1).tolist()
    mean_prob_list = prob_df[BIRD_LIST].mean(axis=1).tolist()
    std_prob_list = prob_df[BIRD_LIST].std(axis=1).tolist()
    max_prob_list = prob_df[BIRD_LIST].max(axis=1).tolist()
    min_prob_list = prob_df[BIRD_LIST].min(axis=1).tolist()
    skew_prob_list = prob_df[BIRD_LIST].skew(axis=1).tolist()
    kurt_prob_list = prob_df[BIRD_LIST].kurt(axis=1).tolist()
    
    X = prob_df[BIRD_LIST].values
    bird_ids_list = np.argsort(-X)[:,:num_candidates]
    row_ids = prob_df["row_id"].tolist()
    rows = [i//num_candidates for i in range(len(bird_ids_list.flatten()))]
    cols = bird_ids_list.flatten()
    # What number?
    ranks = [i%num_candidates for i in range(len(rows))]
    probs_list = X[rows, cols]
    D = {
        "row_id": [row_ids[i] for i in rows],
        "rank": ranks,
        "bird_id": bird_ids_list.flatten(),
        "prob": probs_list.flatten(),
        "sum_prob": [sum_prob_list[i//num_candidates] for i in range(num_candidates*len(mean_prob_list))],
        "mean_prob": [mean_prob_list[i//num_candidates] for i in range(num_candidates*len(mean_prob_list))],
        "std_prob": [std_prob_list[i//num_candidates] for i in range(num_candidates*len(std_prob_list))],
        "max_prob": [max_prob_list[i//num_candidates] for i in range(num_candidates*len(max_prob_list))],
        "min_prob": [min_prob_list[i//num_candidates] for i in range(num_candidates*len(min_prob_list))],
        "skew_prob": [skew_prob_list[i//num_candidates] for i in range(num_candidates*len(skew_prob_list))],
        "kurt_prob": [kurt_prob_list[i//num_candidates] for i in range(num_candidates*len(kurt_prob_list))],
    }
    audio_ids = prob_df["audio_id"].values[rows]
    for diff in range(-num_prob, num_prob+1):
        if diff == 0:
            continue
        neighbor_audio_ids = prob_df["audio_id"].shift(diff).values[rows]
        Y = prob_df[BIRD_LIST].shift(diff).values
        c = f"next{abs(diff)}_prob" if diff < 0 else f"prev{diff}_prob"
        c = c.replace("1_prob", "_prob") # Fix next1_prob to next_prob
        v = Y[rows, cols].flatten()
        v[audio_ids != neighbor_audio_ids] = np.nan
        D[c] = v

    candidate_df = pd.DataFrame(D)
    columns = [
        "row_id",
        "site",
        "year",
        "month",
        "audio_id",
        "seconds",
        "birds",
    ]
    candidate_df = pd.merge(
        candidate_df,
        prob_df[columns],
        how="left",
        on="row_id"
    )
    candidate_df["target"] = candidate_df.apply(
        lambda row: IDX2BIRD[row["bird_id"]] in set(row["birds"].split()),
        axis=1
    )
    candidate_df["label"] = candidate_df["bird_id"].map(IDX2BIRD)
    return candidate_df

def load_metadata():
    meta_df = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
    meta_df["id"] = meta_df.index + 1
    meta_df["year"] = meta_df["date"].apply(lambda _: _.split("-")[0]).astype(int)
    meta_df["month"] = meta_df["date"].apply(lambda _: _.split("-")[1]).astype(int)
    return meta_df


def to_zscore(row):
    x = row["prob"]
    mu = row["prob_avg_in_same_audio"]
    sigma = row["prob_var_in_same_audio"] ** 0.5
    if sigma < 1e-6:
        return 0
    else:
        return (x - mu) / sigma


def add_same_audio_features(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame
):
    # Average probability per bird in the same audio
    _gdf = df.groupby(["audio_id"], as_index=False).mean()[["audio_id"] + BIRD_LIST]
    _df = pd.melt(
        _gdf,
        id_vars=["audio_id"]
    ).rename(columns={
        "variable": "label",
        "value": "prob_avg_in_same_audio"
    })
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
    # Maximum value for each bird in the same audio
    _gdf = df.groupby(["audio_id"], as_index=False).max()[["audio_id"] + BIRD_LIST]
    _df = pd.melt(
        _gdf,
        id_vars=["audio_id"]
    ).rename(columns={
        "variable": "label",
        "value": "prob_max_in_same_audio"
    })
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
    # Variance of each bird in the same audio
    _gdf = df.groupby(["audio_id"], as_index=False).var()[["audio_id"] + BIRD_LIST]
    _df = pd.melt(
        _gdf,
        id_vars=["audio_id"]
    ).rename(columns={
        "variable": "label",
        "value": "prob_var_in_same_audio"
    })
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["audio_id", "label"])
    candidate_df["zscore_in_same_audio"] = candidate_df.apply(to_zscore, axis=1)
    return candidate_df

def add_features(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    max_distance:int,
):
    meta_df = load_metadata()
    # latitude & longitude
    if not "latitude" in candidate_df.columns:
        candidate_df["latitude"] = candidate_df["site"].apply(to_latitude)
    if not "longitude" in candidate_df.columns:
        candidate_df["longitude"] = candidate_df["site"].apply(to_longitude)
    # Number of Appearances
    candidate_df["num_appear"] = candidate_df["label"].map(
        meta_df["primary_label"].value_counts()
    )
    meta_df["site"] = meta_df.apply(
        lambda row: to_site(
            row,
            max_distance=max_distance
        ),
        axis=1
    )

    # Number of occurrences by region
    _df = meta_df.groupby(
        ["primary_label", "site"],
        as_index=False
    )["id"].count().rename(
        columns={
            "primary_label": "label",
            "id": "site_num_appear"
        }
    )
    candidate_df = pd.merge(
        candidate_df,
        _df,
        how="left",
        on=["label", "site"]
    )
    candidate_df["site_appear_ratio"] = candidate_df["site_num_appear"] / candidate_df["num_appear"]
    # Seasonal statistics
    _df = meta_df.groupby(
        ["primary_label", "month"],
        as_index=False
    )["id"].count().rename(
        columns={
            "primary_label": "label",
            "id": "month_num_appear"
        }
    )
    candidate_df = pd.merge(candidate_df, _df, how="left", on=["label", "month"])
    candidate_df["month_appear_ratio"] = candidate_df["month_num_appear"] / candidate_df["num_appear"]

    candidate_df = add_same_audio_features(candidate_df, df)

    # Correction of probability (all down)
    candidate_df["prob / num_appear"] = candidate_df["prob"] / (candidate_df["num_appear"].fillna(0) + 1)
    candidate_df["prob / site_num_appear"] = candidate_df["prob"] / (candidate_df["site_num_appear"].fillna(0) + 1)
    candidate_df["prob * site_appear_ratio"] = candidate_df["prob"] * (candidate_df["site_appear_ratio"].fillna(0) + 0.001)

    # Amount of change from the previous and following frames
    candidate_df["prob_avg"] = candidate_df[["prev_prob", "prob", "next_prob"]].mean(axis=1)
    candidate_df["prob_diff"] = candidate_df["prob"] - candidate_df["prob_avg"]
    candidate_df["prob - prob_max_in_same_audio"] = candidate_df["prob"] - candidate_df["prob_max_in_same_audio"]

    # Average of back and forward frames

    return candidate_df

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):

        melspec = lb.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec

    
def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def crop_or_pad(y, length):
    if len(y) < length:
        y = np.concatenate([y, length - np.zeros(len(y))])
    elif len(y) > length:
        y = y[:length]
    return y


class BirdCLEFDataset(Dataset):
    def __init__(self, data, sr=32_000, n_mels=128, fmin=0, fmax=None, duration=5, step=None, res_type="kaiser_fast", resample=True):

        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        self.npy_save_root = Path("../output/06_stage3/birdclef_dataset")
        
        os.makedirs(self.npy_save_root, exist_ok=True)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        filename = filepath.stem
        npy_path = self.npy_save_root / f"{filename}.npy"
        
        if not os.path.exists(npy_path):
            audio, orig_sr = sf.read(filepath, dtype="float32")

            if self.resample and orig_sr != self.sr:
                audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

            audios = []
            for i in range(self.audio_length, len(audio) + self.step, self.step):
                start = max(0, i - self.audio_length)
                end = start + self.audio_length
                audios.append(audio[start:end])

            if len(audios[-1]) < self.audio_length:
                audios = audios[:-1]

            images = [self.audio_to_image(audio) for audio in audios]
            images = np.stack(images)
            
            np.save(str(npy_path), images)
        return np.load(npy_path)

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])

    
def load_net(checkpoint_path, num_classes=397):
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(DEVICE)
    net = net.eval()
    return net


@torch.no_grad()
def get_thresh_preds(out, thresh=None):
    thresh = thresh or THRESH
    o = (-out).argsort(1)
    npreds = (out > thresh).sum(1)
    preds = []
    for oo, npred in zip(o, npreds):
        preds.append(oo[:npred].cpu().numpy().tolist())
    return preds


def predict(nets, test_data, names=True):
    preds = []
    with torch.no_grad():
        for idx in  tqdm(list(range(len(test_data)))):
            xb = torch.from_numpy(test_data[idx]).to(DEVICE)
            pred = 0.
            for net in nets:
                o = net(xb)
                o = torch.sigmoid(o)
                pred += o
            pred /= len(nets)
            if names:
                pred = BIRD_LIST(get_thresh_preds(pred))

            preds.append(pred)
    return preds

def get_prob_df(config, audio_paths):
    data = pd.DataFrame(
         [(path.stem, *path.stem.split("_"), path) for path in Path(audio_paths).glob("*.ogg")],
        columns = ["filename", "id", "site", "date", "filepath"]
    )
    test_data = BirdCLEFDataset(data=data)

    for checkpoint_path in config.checkpoint_paths:
        prob_filepath = config.get_prob_filepath_from_checkpoint(checkpoint_path)
        if (not os.path.exists(prob_filepath)) or (TARGET_PATH is None):  # Always calculate when no cash is available or when submitting.
            nets = [load_net(checkpoint_path.as_posix())]
            pred_probas = predict(nets, test_data, names=False)
            if TARGET_PATH: # local                
                df = pd.read_csv(TARGET_PATH, usecols=["row_id", "birds"])
            else: # when it is submission
                if str(audio_paths)=="../input/birdclef-2021/train_soundscapes":
                    print(audio_paths)
                    df = pd.read_csv(Path("../input/birdclef-2021/train_soundscape_labels.csv"), usecols=["row_id", "birds"])
                else:
                    print(SAMPLE_SUB_PATH)
                    df = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id", "birds"])
            df["audio_id"] = df["row_id"].apply(lambda _: int(_.split("_")[0]))
            df["site"] = df["row_id"].apply(lambda _: _.split("_")[1])
            df["seconds"] = df["row_id"].apply(lambda _: int(_.split("_")[2]))
            assert len(data) == len(pred_probas)
            n = len(data)
            audio_id_to_date = {}
            audio_id_to_site = {}
            for filepath in audio_paths.glob("*.ogg"):
                audio_id, site, date = os.path.basename(filepath).replace(".ogg", "").split("_")
                audio_id = int(audio_id)
                audio_id_to_date[audio_id] = date
                audio_id_to_site[audio_id] = site
            dfs = []
            for i in range(n):
                row = data.iloc[i]
                audio_id = int(row["id"])
                pred = pred_probas[i]
                _df = pd.DataFrame(pred.to("cpu").numpy())
                _df.columns = [IDX2BIRD[j] for j in range(_df.shape[1])]
                _df["audio_id"] = audio_id
                _df["date"] = audio_id_to_date[audio_id]
                _df["site"] = audio_id_to_site[audio_id]
                _df["seconds"] = [(j+1)*5 for j in range(120)]
                dfs.append(_df)
            prob_df = pd.concat(dfs)
            prob_df = pd.merge(prob_df, df, how="left", on=["site", "audio_id", "seconds"])
            print(f"Save to {prob_filepath}")
            prob_df.to_csv(prob_filepath, index=False)

    # Ensemble
    prob_df = pd.read_csv(
        config.get_prob_filepath_from_checkpoint(config.checkpoint_paths[0])
    )
    if len(config.checkpoint_paths) > 1:
        columns = BIRD_LIST
        for checkpoint_path in config.checkpoint_paths[1:]:
            _df = pd.read_csv(
                config.get_prob_filepath_from_checkpoint(checkpoint_path)
            )
            prob_df[columns] += _df[columns]
        prob_df[columns] /= len(config.checkpoint_paths)

    return prob_df

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(
    candidate_df:pd.DataFrame,
    df:pd.DataFrame,
    candidate_df_soundscapes:pd.DataFrame,
    df_soundscapes:pd.DataFrame,
    num_kfolds:int,
    num_candidates:int,
    verbose:bool=False,
    sampling_strategy:float=1.0,
    random_state:int=777,
):
    
    seed_everything(random_state)
    feature_names = get_feature_names()
    if verbose:
        print("features", feature_names)
        
        
    # short audio の  k fold
    groups = candidate_df["audio_id"]
    kf = StratifiedGroupKFold(n_splits=num_kfolds) # When using lgbm_rank, it is necessary to use the data attached to each group, so don't shuffle them.
    for kfold_index, (_, valid_index) in enumerate(kf.split(candidate_df[feature_names].values, candidate_df["target"].values, groups)):
        candidate_df.loc[valid_index, "fold"] = kfold_index
                        
    X = candidate_df[feature_names].values
    y = candidate_df["target"].values
    oofa = np.zeros(len(candidate_df_soundscapes), dtype=np.float32)
    
    for kfold_index in range(num_kfolds):
        print(f"fold {kfold_index}")
        train_index = candidate_df[candidate_df["fold"] != kfold_index].index
        valid_index = candidate_df[candidate_df["fold"] == kfold_index].index
        X_train, y_train = X[train_index], y[train_index]
        #X_valid, y_valid = X[valid_index], y[valid_index]
        X_valid, y_valid = candidate_df_soundscapes[feature_names].values, candidate_df_soundscapes["target"].values
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            # 'device':'gpu', TODO change to GPU for cluster
            'device':'cpu'
        }
        model = lgb.train(
            params,
            dtrain,
            valid_sets=dvalid,
            num_boost_round=200,
            early_stopping_rounds=20,
            verbose_eval=20,
        )
        oofa += model.predict(X_valid.astype(np.float32))/num_kfolds
        pickle.dump(model, open(LGBM_PKL_DIR + f"lgbm_{kfold_index}.pkl", "wb"))
        
    def f(th):
        _df = candidate_df_soundscapes[(oofa > th)]
        if len(_df) == 0:
            return 0
        _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["label"].apply(lambda _: " ".join(_))
        df2 = pd.merge(
            df_soundscapes[["audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
        df2.loc[df2["label"].isnull(), "label"] = "nocall"
        return df2.apply(
            lambda _: get_metrics(_["birds"], _["label"])["f1"],
            axis=1
        ).mean()


    print("-"*30)
    print(f"#sound_scapes (len:{len(candidate_df_soundscapes)}) でのスコア")
    lb, ub = 0, 1
    for k in range(30):
        th1 = (2*lb + ub) / 3
        th2 = (lb + 2*ub) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("best th: %.4f" % th)
    print("best F1: %.4f" % f(th))
    if verbose:
        y_soundscapes =  candidate_df_soundscapes["target"].values
        oof = (oofa > th).astype(int)
        print("[details] Call or No call classirication")
        print("binary F1: %.4f" % f1_score(y_soundscapes, oof))
        print("gt positive ratio: %.4f" % np.mean(y_soundscapes))
        print("oof positive ratio: %.4f" % np.mean(oof))
        print("Accuracy: %.4f" % accuracy_score(y_soundscapes, oof))
        print("Recall: %.4f" % recall_score(y_soundscapes, oof))
        print("Precision: %.4f" % precision_score(y_soundscapes, oof))
    print("-"*30)
    print()

def get_feature_names() -> List[str]:
    return [
        "year",
        "month",
        "sum_prob",
        "mean_prob",
        #"std_prob",
        "max_prob",
        #"min_prob",
        #"skew_prob",
        #"kurt_prob",
        "prev6_prob",
        "prev5_prob",
        "prev4_prob",
        "prev3_prob",
        "prev2_prob",
        "prev_prob",
        "prob",
        "next_prob",
        "next2_prob",
        "next3_prob",
        "next4_prob",
        "next5_prob",
        "next6_prob",
        "rank",
        "latitude",
        "longitude",
        "bird_id", # +0.013700
        "seconds", # -0.0050
        "num_appear",
        "site_num_appear",
        "site_appear_ratio",
        # "prob / num_appear", # -0.005
        # "prob / site_num_appear", # -0.0102
        # "prob * site_appear_ratio", # -0.0049
        # "prob_avg", # -0.0155
        "prob_diff", # 0.0082
        # "prob_avg_in_same_audio", # -0.0256
        # "prob_max_in_same_audio", # -0.0142
        # "prob_var_in_same_audio", # -0.0304
        # "prob - prob_max_in_same_audio", # -0.0069
        # "zscore_in_same_audio", # -0.0110
        # "month_num_appear", # 0.0164
    ]


def get_metrics(s_true, s_pred):
    s_true = set(s_true.split())
    s_pred = set(s_pred.split())
    n, n_true, n_pred = len(s_true.intersection(s_pred)), len(s_true), len(s_pred)
    prec = n/n_pred
    rec = n/n_true
    f1 = 2*prec*rec/(prec + rec) if prec + rec else 0
    return {
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "n_true": n_true,
        "n_pred": n_pred,
        "n": n
    }

def optimize(
    candidate_df:pd.DataFrame,
    prob_df:pd.DataFrame,
    num_kfolds:int,
    weights_filepath_dict:dict,
):
    feature_names = get_feature_names()
    X = candidate_df[feature_names].values
    y_preda_list = []
    for mode in weights_filepath_dict.keys():
        fold_y_preda_list = []
        for kfold_index in range(num_kfolds):
            clf = pickle.load(open(weights_filepath_dict[mode][kfold_index], "rb"))
            if mode=='lgbm':
                y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
            elif mode=='lgbm_rank':
                y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
            else:
                y_preda = clf.predict_proba(X)[:,1]
            fold_y_preda_list.append(y_preda)
        mean_preda = np.mean(fold_y_preda_list, axis=0)
        if mode=='lgbm_rank': # scaling
            mean_preda = 1/(1 + np.exp(-mean_preda))
        y_preda_list.append(mean_preda)
    y_preda = np.mean(y_preda_list, axis=0)
    candidate_df["y_preda"] = y_preda
    
    def f(th):
        _df = candidate_df[y_preda > th]
        if len(_df) == 0:
            return 0
        _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
        )["label"].apply(
            lambda _: " ".join(_)
        ).rename(columns={
            "label": "predictions"
        })
        submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
        submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"
        return submission_df.apply(
            lambda row: get_metrics(row["birds"], row["predictions"])["f1"],
            axis=1
        ).mean()
    
    lb, ub = 0, 1
    for k in range(30):
        th1 = (lb * 2 + ub) / 3
        th2 = (lb + ub * 2) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("-" * 30)
    print("📌best threshold: %f" % th)
    print("best F1: %f" % f(th))
    
    # nocall injection
    _df = candidate_df[y_preda > th]
    if len(_df) == 0:
        return 0
    _gdf = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
    )["label"].apply(
        lambda _: " ".join(_)
    ).rename(columns={
        "label": "predictions"
    })
    submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
    submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"

    
    _gdf2 = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
    )["y_preda"].sum()
    submission_df = pd.merge(
            submission_df,
            _gdf2,
            how="left",
            on=["audio_id", "seconds"]
        )
    def f_nocall(nocall_th):
        submission_df_with_nocall = submission_df.copy()
        submission_df_with_nocall.loc[(submission_df_with_nocall["y_preda"]<nocall_th) 
                                      & (submission_df_with_nocall["predictions"]!="nocall"), "predictions"] += " nocall"
        return submission_df_with_nocall.apply(
            lambda row: get_metrics(row["birds"], row["predictions"])["f1"],
            axis=1
        ).mean()
    lb, ub = 0, 1
    for k in range(30):
        th1 = (lb * 2 + ub) / 3
        th2 = (lb + ub * 2) / 3
        if f_nocall(th1) < f_nocall(th2):
            lb = th1
        else:
            ub = th2
    nocall_th = (lb + ub) / 2
    print("-" * 30)
    print("## nocall injection")
    print("📌best nocall threshold: %f" % nocall_th)
    print("best F1: %f" % f_nocall(nocall_th))
    
    return th, nocall_th

def calc_baseline(prob_df:pd.DataFrame):
    """Calculate the optimal value of F1 score simply based on the threshold alone (without 3rd stage)"""
    columns = BIRD_LIST
    X = prob_df[columns].values
    def f(th):
        n = X.shape[0]
        pred_labels = [[] for i in range(n)]
        I, J = np.where(X > th)
        for i, j in zip(I, J):
            pred_labels[i].append(IDX2BIRD[j])
        for i in range(n):
            if len(pred_labels[i]) == 0:
                pred_labels[i] = "nocall"
            else:
                pred_labels[i] = " ".join(pred_labels[i])
        prob_df["pred_labels"] = pred_labels
        return prob_df.apply(
            lambda _: get_metrics(_["birds"], _["pred_labels"])["f1"],
            axis=1
        ).mean()

    lb, ub = 0, 1
    for k in range(30):
        th1 = (2*lb + ub) / 3
        th2 = (lb + 2*ub) / 3
        if f(th1) < f(th2):
            lb = th1
        else:
            ub = th2
    th = (lb + ub) / 2
    print("best th: %.4f" % th)
    print("best F1: %.4f" % f(th))
    return th

def make_submission(
    candidate_df:pd.DataFrame,
    prob_df:pd.DataFrame,
    num_kfolds:int,
    th:float,
    nocall_th:float,
    weights_filepath_dict:dict,
    max_distance:int
):
    feature_names = get_feature_names()
    X = candidate_df[feature_names].values
    y_preda_list = []
    for mode in weights_filepath_dict.keys():
        fold_y_preda_list = []
        for kfold_index in range(num_kfolds):
            clf = pickle.load(open(weights_filepath_dict[mode][kfold_index], "rb"))
            if mode=='lgbm':
                y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
            elif mode=='lgbm_rank':
                y_preda = clf.predict(X.astype(np.float32), num_iteration=clf.best_iteration)
            else:
                y_preda = clf.predict_proba(X)[:,1]
            fold_y_preda_list.append(y_preda)
        mean_preda = np.mean(fold_y_preda_list, axis=0)
        if mode=='lgbm_rank':  # scaling
            mean_preda = 1/(1 + np.exp(-mean_preda))
        y_preda_list.append(mean_preda)
    y_preda = np.mean(y_preda_list, axis=0)
    candidate_df["y_preda"] = y_preda
    
    _df = candidate_df[y_preda > th]
    _gdf = _df.groupby(
        ["audio_id", "seconds"],
        as_index=False
    )["label"].apply(
        lambda _: " ".join(_)
    ).rename(columns={
        "label": "predictions"
    })
    submission_df = pd.merge(
            prob_df[["row_id", "audio_id", "seconds", "birds"]],
            _gdf,
            how="left",
            on=["audio_id", "seconds"]
        )
    submission_df.loc[submission_df["predictions"].isnull(), "predictions"] = "nocall"
    if TARGET_PATH:
        score_df = pd.DataFrame(
            submission_df.apply(
                lambda row: get_metrics(row["birds"], row["predictions"]),
                axis=1
            ).tolist()
        )
        print("-" * 30)
        print("BEFORE nocall injection")
        print("CV score on a trained model with train_short_audio (to check the model behavior)")
        print("F1: %.4f" % score_df["f1"].mean())
        print("Recall: %.4f" % score_df["rec"].mean())
        print("Precision: %.4f" % score_df["prec"].mean())
        
    # nocall injection
    _gdf2 = _df.groupby(
            ["audio_id", "seconds"],
            as_index=False
    )["y_preda"].sum()
    submission_df = pd.merge(
            submission_df,
            _gdf2,
            how="left",
            on=["audio_id", "seconds"]
        )
    submission_df.loc[(submission_df["y_preda"] < nocall_th) 
                     & (submission_df["predictions"]!="nocall"), "predictions"] += " nocall"
    if TARGET_PATH:
        score_df = pd.DataFrame(
            submission_df.apply(
                lambda row: get_metrics(row["birds"], row["predictions"]),
                axis=1
            ).tolist()
        )
        print("-" * 30)
        print("AFTER nocall injection")
        print("CV score on a trained model with train_short_audio (to check the model behavior)")
        print("F1: %.4f" % score_df["f1"].mean())
        print("Recall: %.4f" % score_df["rec"].mean())
        print("Precision: %.4f" % score_df["prec"].mean())
                
    return submission_df[["row_id", "predictions"]].rename(columns={
        "predictions": "birds"
    })

####################################################
# Train the model for the table competition part.
####################################################

TEST_AUDIO_ROOT = Path("../input/birdclef-2021/test_soundscapes")
SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
TARGET_PATH = None

if not len(list(TEST_AUDIO_ROOT.glob("*.ogg"))): # If there isn't any sound source for testing, call for train_soundscapes
    TEST_AUDIO_ROOT = Path("../input/birdclef-2021/train_soundscapes")
    SAMPLE_SUB_PATH = None
    # SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
    TARGET_PATH = Path("../input/birdclef-2021/train_soundscape_labels.csv")


# short audio
# Exclude items that do not need to be trained
if not "site" in prob_df.columns:
    prob_df["site"] = prob_df.apply(
        lambda row: to_site(
            row,
            max_distance=training_config.max_distance
        ),
        axis=1
    )
    print("[exclude other]before: %d" % len(prob_df))
    prob_df = prob_df[prob_df["site"] != "Other"].reset_index(drop=True)
    print("[exclude other]after: %d" % len(prob_df))


candidate_df = make_candidates(
    prob_df,
    num_spieces=training_config.num_spieces,
    num_candidates=training_config.num_candidates,
    max_distance=training_config.max_distance
)
candidate_df = add_features(
    candidate_df,
    prob_df,
    max_distance=training_config.max_distance
)
   
# soundscapes
prob_df_soundscapes = get_prob_df(config,Path("../input/birdclef-2021/train_soundscapes"))
candidate_df_soundscapes = make_candidates(
    prob_df_soundscapes,
    num_spieces=training_config.num_spieces,
    num_candidates=training_config.num_candidates,
    max_distance=training_config.max_distance
)
candidate_df_soundscapes = add_features(
    candidate_df_soundscapes,
    prob_df_soundscapes,
    max_distance=training_config.max_distance
)    
    
    
for mode in config.weights_filepath_dict.keys():
    print(f'training of {mode} is going...')
    train(
        candidate_df,
        prob_df,
        candidate_df_soundscapes,
        prob_df_soundscapes,
        num_kfolds=training_config.num_kfolds,
        num_candidates=training_config.num_candidates,
        verbose=True,
    )

######################################################
# for submission
######################################################
prob_df = get_prob_df(config, TEST_AUDIO_ROOT)

# candidate extraction
candidate_df = make_candidates(
    prob_df,
    num_spieces=config.num_spieces,
    num_candidates=config.num_candidates,
    max_distance=config.max_distance,
    num_prob=config.num_prob,
    nocall_threshold=config.nocall_threshold
)
# add features
candidate_df = add_features(
    candidate_df,
    prob_df,
    max_distance=config.max_distance
)
    
if TARGET_PATH:
    best_th, best_nocall_th = optimize(
        candidate_df,
        prob_df,
        num_kfolds=config.num_kfolds,
        weights_filepath_dict=config.weights_filepath_dict,
    )
if config.check_baseline:
    print("-" * 30)
    print("check F1 score without 3rd stage(table competition)")
    calc_baseline(prob_df)

submission_df = make_submission(
    candidate_df,
    prob_df,
    num_kfolds=config.num_kfolds,
    th=best_th,
    nocall_th=best_nocall_th,
    weights_filepath_dict=config.weights_filepath_dict,
    max_distance=config.max_distance
)

submission_df.to_csv("../output/submission.csv", index=False)

