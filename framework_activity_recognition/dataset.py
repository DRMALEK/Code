import os
import torch
import random as rn
import numpy as np
import logging
from torch.utils.data import Dataset
from PIL import Image
import math
from framework_activity_recognition import transform



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval. If the number of frames is < num_samples, duplicate frames.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """


    length = (end_idx - start_idx)+1
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, end_idx).long()
    index = index-start_idx

    out_frames = torch.index_select(frames, 0, index)

    return out_frames

class MeccanoDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for the MECCANO dataset used for activity recognition.

    Args:
        cfg (dict): Configuration dictionary containing dataset and training/testing parameters.
        mode (str): Mode in which the dataset is used. Must be one of ["train", "val", "test"].
        num_retries (int): Number of retries for loading a video. Default is 10.

    Attributes:
        mode (str): Mode in which the dataset is used.
        cfg (dict): Configuration dictionary.
        _num_retries (int): Number of retries for loading a video.
        _num_clips (int): Number of clips per video.
        _path_to_videos (list): List of paths to video files.
        _labels (list): List of labels for each video.
        _spatial_temporal_idx (list): List of spatial-temporal indices for each video.
        _frame_start (list): List of start frames for each video.
        _frame_end (list): List of end frames for each video.

    Methods:
        _construct_loader(): Constructs the data loader by reading video paths and labels from a CSV file.
        __getitem__(index): Returns a tuple containing the frames, label, index, and an empty dictionary for a given index.
        __len__(): Returns the number of videos in the dataset.
        spatial_sampling(frames, spatial_idx, min_scale, max_scale, crop_size, random_horizontal_flip):
            Performs spatial sampling on the given frames.
    """
    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], "Split '{}' not supported for MECCANO".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._num_retries = num_retries
        self._num_clips = 1
     
        logger.info("Constructing MECCANO {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        path_to_file = os.path.join(
            self.cfg["data"]["path_to_data_dir"], "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_start = []
        self._frame_end = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if clip_idx == 0:
                    continue

                assert len(path_label.split(',')) == 5
                video_path, action_label, action_noun, frame_start, frame_end = path_label.split(',')
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg["data"]["path_to_data_dir"], self.mode, video_path)
                    )
                    self._frame_start.append(frame_start)
                    self._frame_end.append(frame_end)
                    self._labels.append(int(action_label))
                    self._spatial_temporal_idx.append(idx)
                
        assert len(self._path_to_videos) > 0, "Failed to load MECCANO split {} from {}".format(self.mode, path_to_file)
        logger.info(
            "Constructing MECCANO dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            temporal_sample_index = -1
            spatial_sample_index = -1 # random crop
            min_scale = self.cfg["data"]["train_jitter_scales"][0]
            max_scale = self.cfg["data"]["train_jitter_scales"][1]
            crop_size = self.cfg["data"]["train_crop_size"]
        
        elif self.mode in ["test"]:
            spatial_sample_index = 1 # center crop
            min_scale, max_scale, crop_size = [self.cfg["data"]["test_crop_size"]] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
        
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        # 
        frames = []
        frame_count = int(self._frame_start[index][:-4])
        while frame_count <= int(self._frame_end[index][:-4]):
            name_frame = str(frame_count).zfill(5)
            image = Image.open(os.path.join(self.cfg["data"]["path_to_data_dir"], "frames", self._path_to_videos[index], name_frame + ".jpg"))
            image = np.asarray(image)
            frames.append(torch.from_numpy(image))
            frame_count += 1
        frames = torch.stack(frames)
        frames = temporal_sampling(frames, int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]), self.cfg["data"]["num_frames"])

        # Normalize the frames by subtracting the mean and dividing by the standard deviation of the dataset
        frames = frames / 255.0
        #frames = frames - torch.tensor(self.cfg["DATA"]["MEAN"])
        #frames = frames / torch.tensor(self.cfg["DATA"]["STD"])
        
        # Transpose the frames to the correct format which is (channel or pixel value, frame number, height, width)
        frames = frames.permute(3, 0, 1, 2)

        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )
        
        label = self._labels[index]
        frames = [frames]
        
        
        return frames, label, index, {}

    def __len__(self):
        return len(self._path_to_videos)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            
            # Randomly scale the frames with short side uniformly sampled from [min_scale, max_scale]
            frames, _ = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
            )
            
            # Randomly crop the frames
            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            assert len({min_scale, max_scale, crop_size}) == 1
            
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )

            # Perform center crop
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
