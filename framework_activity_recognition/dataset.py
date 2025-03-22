import os
import torch
import numpy as np
import logging
from PIL import Image

from framework_activity_recognition.sampling import temporal_sampling, spatial_sampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


classes_name = [
    'check_booklet', 'take_gray_perforated_bar', 'take_gray_angled_perforated_bar', 'align_objects', 'take_screw',
    'plug_screw', 'take_bolt', 'tighten_bolt_with_hands', 'unscrew_screw_with_hands', 'pull_screw',
    'put_gray_angled_perforated_bar', 'take_white_angled_perforated_bar', 'take_handlebar', 'plug_handlebar', 'pull_partial_model',
    'put_partial_model', 'put_white_angled_perforated_bar', 'take_red_angled_perforated_bar', 'take_red_perforated_bar',
    'put_red_perforated_bar', 'put_screw', 'take_partial_model', 'take_red_4_perforated_junction_bar', 'take_objects', 'take_screwdriver',
    'screw_screw_with_screwdriver', 'align_screwdriver_to_screw', 'put_screwdriver', 'take_washer', 'take_wheels_axle', 'take_tire', 'take_rim',
    'fit_rim_tire', 'take_rod', 'put_rod', 'put_washer', 'plug_rod', 'take_roller', 'browse_booklet', 'put_gray_perforated_bar',
    'put_red_4_perforated_junction_bar', 'take_booklet', 'put_booklet', 'put_bolt', 'put_wheels_axle', 'take_red_perforated_junction_bar',
    'put_tire', 'put_rim', 'pull_rod', 'put_roller', 'put_red_perforated_junction_bar', 'put_objects', 'take_wrench',
    'put_wrench', 'align_wrench_to_bolt', 'tighten_bolt_with_wrench', 'unscrew_screw_with_screwdriver', 'put_red_angled_perforated_bar',
    'loosen_bolt_with_hands', 'put_handlebar', 'screw_screw_with_hands'
]

class MeccanoDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for the MECCANO dataset used for activity recognition.

    Args:quantization_framework
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
    """
    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], "Split '{}' not supported for MECCANO".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._num_retries = num_retries
        self._num_clips = 1
        self.annotation_converter = classes_name
 
        self.nClasses = len(self.annotation_converter)

        logger.info("Constructing MECCANO {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        path_to_csv_file = os.path.join(
            self.cfg["data"]["path_to_data_dir"], "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_csv_file), "{} dir not found".format(path_to_csv_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_start = []
        self._frame_end = []
        with open(path_to_csv_file, "r") as f:
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
                
        assert len(self._path_to_videos) > 0, "Failed to load MECCANO split {} from {}".format(self.mode, path_to_csv_file)
        logger.info(
            "Constructing MECCANO dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_csv_file
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

        # Recover the frames from the video
        frames = []
        frame_count = int(self._frame_start[index][:-4])
        while frame_count <= int(self._frame_end[index][:-4]):
            name_frame = str(frame_count).zfill(5)
            image = Image.open(os.path.join(self._path_to_videos[index], name_frame + ".jpg"))
            image = np.asarray(image)
            frames.append(torch.from_numpy(image))
            frame_count += 1
        frames = torch.stack(frames)
        frames = temporal_sampling(frames, int(self._frame_start[index][:-4]), int(self._frame_end[index][:-4]), self.cfg["data"]["num_frames"])

        # Normalize the frames to [-1, 1]
        #frames = (frames / 255.0) * 2 - 1
        
        # Normalize the frames to [0, 1]
        frames = frames / 255.0

        # Use kinetics 400 mean and std values for normalization
        frames = frames - torch.tensor([0.45, 0.45, 0.45])
        frames = frames / torch.tensor([0.225, 0.225, 0.225])

        
        #Mean and std values for the dataset (using only validaiton set)
        #Mean: [0.4291935919783522, 0.4138912852383532, 0.3932020827306195]
        #Std: [0.2106381314194434, 0.21905233733269974, 0.23907043718073798]
        
        # Transpose the frames to the correct format which is (channel or pixel value, frame number, height, width)
        frames = frames.permute(3, 0, 1, 2)


        frames = spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )
        
        label = self._labels[index]
        
        #return frames, label, index, {}
        return frames, label

    def __len__(self):
        return len(self._path_to_videos)

    def get_labels(self):
        return self._labels