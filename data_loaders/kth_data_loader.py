import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
import imageio
import cv2
from utils.utils import makedir
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb

DEBUG = False


class KTHDataLoader:
    def __init__(self, args):
        if args.dataset == 'KTH':
            # Data Loading
            kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
            # TODO; add on textroot, dataroot, image_size
            dataset = KTH_Dataset(args.dataroot, os.path.join(args.textroot, 'train_list.txt'), 16, args.input_shape,
                                  1, True)
            video_dataset = KTHImgDataset(dataset, 16, every_nth=1)
            self.train_loader = DataLoader(video_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

            # get the validate dataloader
            val_dataset = KTH_Dataset(args.dataroot, os.path.join(args.textroot, 'val_list.txt'), 16, args.input_shape,
                                      1, True)
            val_video_dataset = KTHImgDataset(val_dataset, 16, every_nth=1)
            self.test_loader = DataLoader(val_video_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
            self.classes = ['boxing', 'handwaving', 'handclapping', 'running', 'jogging', 'walking']

        else:
            raise ValueError('The dataset should be CIFAR10')


class KTH_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, videolist, video_len, input_shape, every_nth, crop):
        self.dataroot = dataroot
        with open(videolist, 'r') as f:
            self.lines = f.readlines()
        self.video_len = video_len
        self.every_nth = every_nth
        self.crop = crop
        self.classes = ['boxing', 'handwaving', 'handclapping', 'running', 'jogging', 'walking']
        self.image_size = input_shape.width
        self.lengths = []
        self.cases = []
        self.cacheroot = os.path.join(self.dataroot, 'npy_%s' % self.image_size)
        makedir(self.cacheroot)
        cache = os.path.join(self.cacheroot, 'cache_%s.db' % videolist.split('/')[-1].split('_')[0])
        if cache is not None and os.path.exists(cache):
            with open(cache, 'r') as f:
                self.cases, self.lengths = pickle.load(f)
        else:
            for idx, line in enumerate(
                    tqdm.tqdm(self.lines, desc="Counting total number of frames")):
                video_name, start_idx, end_idx = line.split()
                start_idx, end_idx = int(start_idx), int(end_idx)
                if end_idx - start_idx > video_len * every_nth:
                    self.lengths.append(end_idx - start_idx + 1)
                    self.cases.append(line)
                    video_path = os.path.join(self.dataroot, video_name + '_uncomp.avi')
                    video = self.load_video(video_path, start_idx - 1, end_idx - 1)
                    np.save(os.path.join(self.cacheroot, video_name + '_%d_%d.npy' % (start_idx, end_idx)), video)
            if cache is not None:
                with open(cache, 'w') as f:
                    pickle.dump((self.cases, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print "Total number of frames {}".format(np.sum(self.lengths))

    def load_video(self, video_path, start_idx, end_idx):
        # open the video
        vid = imageio.get_reader(video_path, 'ffmpeg')
        # read in videos
        frames = []
        for idx in range(start_idx, end_idx + 1):
            try:
                img = np.array(vid.get_data(idx))
            except IndexError:
                print('video length = %d' % vid.get_length())
                print('idx = %d in %s does not exist' % (idx, video_path))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            else:
                img = img[:, :, ::-1]
            frames.append(np.array(img))
        video = np.stack(frames, axis=0)
        return video

    def read_seq(self, video_path):
        # open the video
        video = np.load(video_path)
        # read in videos
        frames = []
        for idx in range(len(video)):
            img = video[idx]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            else:
                img = img[:, :, ::-1]
            frames.append(np.array(img))

        # img_size
        h, w, _ = frames[-1].shape

        # random crop
        if self.crop:
            h_start = np.random.randint(low=0, high=10)
            w_start = np.random.randint(low=0, high=10)
        else:
            h_start = 5
            w_start = 5

        for idx, frame in enumerate(frames):
            frames[idx] = frame[h_start: h_start + (h - 10), w_start: w_start + (w - 10)]

        for idx, frame in enumerate(frames):
            frames[idx] = cv2.resize(frame, (self.image_size, self.image_size))

        return frames

    def __getitem__(self, item):
        video_name, start_idx, end_idx = self.cases[item].split()
        start_idx, end_idx = int(start_idx), int(end_idx)
        video_path = os.path.join(self.cacheroot, video_name + '_%d_%d.npy' % (start_idx, end_idx))
        frames = self.read_seq(video_path)
        video = np.concatenate(frames, axis=1)
        label = self.classes.index(video_name.split('_')[1])
        return video, label

    def __len__(self):
        return len(self.cases)


class KTHImgDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x[0, ::].unsqueeze(dim=0),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # TODO check the input range

    def __getitem__(self, item):
        """
        :param item: the query index
        :return: 'images': a tensor of [c, video_length, h, w], 'categories': a int, class label
        """
        video, target = self.dataset[item]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        video_len = longer // shorter

        # videos can be of various length, we randomly sample sub-sequences
        idx = np.random.randint(0, video_len-1)
        frames = np.split(video, video_len, axis=1 if horizontal else 0)
        selected = frames[idx]
        return self.transforms(selected), target

    def __len__(self):
        return len(self.dataset)
