#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import numpy as np
import math

def random_short_side_scale_jitter(
        images, min_size, max_size, boxes=None, inverse_uniform_sampling=False
    ):
        """
        Perform a spatial short scale jittering on the given images and
        corresponding boxes.
        Args:
            images (tensor): images to perform scale jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
            min_size (int): the minimal size to scale the frames.
            max_size (int): the maximal size to scale the frames.
            boxes (ndarray): optional. Corresponding boxes to images.
                Dimension is `num boxes` x 4.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale, max_scale].
        Returns:
            (tensor): the scaled images with dimension of
                `num frames` x `channel` x `new height` x `new width`.
            (ndarray or None): the scaled boxes with dimension of
                `num boxes` x 4.
        """
        if inverse_uniform_sampling:
            size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))
        else:
            size = int(round(np.random.uniform(min_size, max_size)))

        height = images.shape[2]
        width = images.shape[3]
        if (width <= height and width == size) or (height <= width and height == size):
            return images, boxes
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
            if boxes is not None:
                boxes = boxes * float(new_height) / height
        else:
            new_width = int(math.floor((float(width) / height) * size))
            if boxes is not None:
                boxes = boxes * float(new_width) / width

        return (
            torch.nn.functional.interpolate(
                images,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ),
            boxes,
        )

def crop_boxes(boxes, x_offset, y_offset):
        """
        Peform crop on the bounding boxes given the offsets.
        Args:
            boxes (ndarray or None): bounding boxes to peform crop. The dimension
                is `num boxes` x 4.
            x_offset (int): cropping offset in the x axis.
            y_offset (int): cropping offset in the y axis.
        Returns:
            cropped_boxes (ndarray or None): the cropped boxes with dimension of
                `num boxes` x 4.
        """
        cropped_boxes = boxes.copy()
        cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
        cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

        return cropped_boxes

def random_crop(images, size, boxes=None):
        """
        Perform random spatial crop on the given images and corresponding boxes.
        Args:
            images (tensor): images to perform random crop. The dimension is
                `num frames` x `channel` x `height` x `width`.
            size (int): the size of height and width to crop on the image.
            boxes (ndarray or None): optional. Corresponding boxes to images.
                Dimension is `num boxes` x 4.
        Returns:
            cropped (tensor): cropped images with dimension of
                `num frames` x `channel` x `size` x `size`.
            cropped_boxes (ndarray or None): the cropped boxes with dimension of
                `num boxes` x 4.
        """
        if images.shape[2] == size and images.shape[3] == size:
            return images, boxes
        height = images.shape[2]
        width = images.shape[3]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

        cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None

        return cropped, cropped_boxes

def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes

def horizontal_flip(prob, images, boxes=None):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
        flipped_boxes (ndarray or None): the flipped boxes with dimension of
            `num boxes` x 4.
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension does not supported")
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes
