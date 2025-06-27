import numpy as np
import cv2
from path import Path
import random
import os 
import pickle
import torch


def load_as_float(path):
    """Loads image"""
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) 
    return im


class NewSequenceFolder(torch.utils.data.Dataset):
    """Creates a dataloader for the new dataset structure"""

    def __init__(self, root, seed=None, ttype='test.txt', sequence_length=2, sequence_gap=20, transform=None, height=240, width=320):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        
        # 读取内参文件
        intrinsics_path = os.path.join(self.root, "intrinsic", "intrinsics.txt")
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
            fx = float(lines[0].split()[0])
            fy = float(lines[1].split()[1])
            cx = float(lines[0].split()[2])
            cy = float(lines[1].split()[1])
            
        self.intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]).astype(np.float32)
        
        # 获取所有图像文件
        color_dir = os.path.join(self.root, "color")
        depth_dir = os.path.join(self.root, "depth")
        pose_files = os.listdir(os.path.join(self.root, "pose"))
        
        # 选择一个姿态文件
        pose_file = pose_files[0]  # 使用第一个姿态文件
        pose_path = os.path.join(self.root, "pose", pose_file)
        
        # 读取所有姿态
        poses_data = np.loadtxt(pose_path)
        poses = []
        for i in range(0, len(poses_data), 4):
            if i + 3 < len(poses_data):
                pose = np.vstack((poses_data[i:i+3], [0, 0, 0, 1]))
                poses.append(pose)
        
        # 获取所有图像文件名
        color_files = sorted(os.listdir(color_dir))
        depth_files = sorted(os.listdir(depth_dir))
        
        # 确保文件数量匹配
        min_frames = min(len(color_files), len(depth_files), len(poses))
        
        self.width = width
        self.height = height
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequence_gap = sequence_gap
        
        # 创建样本序列
        self.samples = []
        for i in range(min_frames):
            if i + (sequence_length-1)*sequence_gap >= min_frames:
                continue
                
            sample = {
                'intrinsics': self.intrinsics,
                'tgt': os.path.join(color_dir, color_files[i]),
                'tgt_depth': os.path.join(depth_dir, depth_files[i]),
                'ref_imgs': [],
                'ref_depths': [],
                'ref_poses': [],
                'path': os.path.join(self.root, color_files[i][:-4])
            }
            
            # 添加参考帧
            for j in range(1, sequence_length):
                ref_idx = i + j * sequence_gap
                if ref_idx < min_frames:
                    sample['ref_imgs'].append(os.path.join(color_dir, color_files[ref_idx]))
                    sample['ref_depths'].append(os.path.join(depth_dir, depth_files[ref_idx]))
                    
                    # 计算相对姿态
                    pose_tgt = poses[i]
                    pose_src = poses[ref_idx]
                    pose_rel = np.linalg.inv(pose_src) @ pose_tgt
                    pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
                    sample['ref_poses'].append(pose)
            
            if len(sample['ref_imgs']) == sequence_length - 1:
                self.samples.append(sample)
        
        # 创建虚拟场景列表，以兼容原代码
        self.scenes = [self.root]

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        tgt_depth = cv2.imread(sample['tgt_depth'], -1).astype(np.float32)/1000

        ref_poses = sample['ref_poses']
        ref_depths = [cv2.imread(depth_img, -1).astype(np.float32)/1000 for depth_img in sample['ref_depths']]
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.transform is not None:
            imgs, depths, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth] + ref_depths, np.copy(sample['intrinsics']), self.height, self.width)
            tgt_img = imgs[0]     
            tgt_depth = depths[0]
            ref_imgs = imgs[1:]
            ref_depths = depths[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
            
        return tgt_img, ref_imgs, ref_poses, intrinsics, tgt_depth, ref_depths

    def __len__(self):
        return len(self.samples)