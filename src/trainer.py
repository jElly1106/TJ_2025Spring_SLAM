import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from path import Path
import scipy.io
import cv2
from pathlib import Path
import shutil  # 添加这一行

from assets.sequence_folders import SequenceFolder
from models import superpoint, triangulation, densedepth
from assets.utils import *



parser = argparse.ArgumentParser(description='DELTAS训练 - scannet数据集',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 数据集参数
parser.add_argument('--data', default='path/to/scannet', type=str, metavar='DIR',
                    help='scannet数据集路径')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='数据集格式')
parser.add_argument('--train-file', default='scannet_train.txt', type=str,
                    help='训练数据列表文件')
parser.add_argument('--val-file', default='scannet_val.txt', type=str,
                    help='验证数据列表文件')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='数据加载工作线程数')
parser.add_argument('-b', '--batch-size', default=6, type=int, metavar='N',
                    help='mini-batch大小')

# 模型参数
parser.add_argument('--mindepth', type=float, default=0.5, help='最小深度')
parser.add_argument('--maxdepth', type=float, default=10.0, help='最大深度')
parser.add_argument('--width', type=int, default=320, help='图像宽度')
parser.add_argument('--height', type=int, default=240, help='图像高度')
parser.add_argument('--seq_length', default=3, type=int, help='序列长度')
parser.add_argument('--seq_gap', default=1, type=int, help='帧间隔')
parser.add_argument('--model_type', type=str, default='resnet50', help='网络骨干')
parser.add_argument('--num_kps', default=256, type=int, help='兴趣点数量')
parser.add_argument('--descriptor_dim', type=int, default=128, help='描述子维度')
parser.add_argument('--detection_threshold', type=float, default=0.0005, 
                    help='兴趣点检测阈值')
parser.add_argument('--frac_superpoint', type=float, default=0.5, help='兴趣点比例')
parser.add_argument('--nms_radius', type=int, default=9, help='非极大值抑制半径')
parser.add_argument('--align_corners', type=bool, default=False, help='对齐角点')

# 三角测量参数
parser.add_argument('--do_confidence', type=bool, default=True, help='三角测量中的置信度')
parser.add_argument('--dist_orthogonal', type=int, default=1, help='像素偏移距离')
parser.add_argument('--kernel_size', type=int, default=1, help='核大小')
parser.add_argument('--out_length', type=int, default=25, help='极线块的输出长度')
parser.add_argument('--depth_range', type=bool, default=True, help='使用深度范围进行限制')

# 训练参数
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='训练轮数')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR',
                    help='初始学习率')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='权重衰减')
parser.add_argument('--print-freq', default=50, type=int, metavar='N',
                    help='打印频率')
parser.add_argument('--save-freq', default=3, type=int, metavar='N',
                    help='保存频率')
parser.add_argument('--seed', default=1, type=int, help='随机种子')
parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点')
parser.add_argument('--pretrained', dest='pretrained', default='', metavar='PATH',
                    help='预训练模型路径')

# 损失函数权重
parser.add_argument('--w_ip', type=float, default=0.1, help='兴趣点检测损失权重')
parser.add_argument('--w_2d', type=float, default=1.0, help='2D匹配损失权重')
parser.add_argument('--w_3d', type=float, default=2.0, help='3D三角测量损失权重')
parser.add_argument('--w_sm', type=float, default=1.0, help='平滑损失权重')
parser.add_argument('--w_d', type=float, nargs='+', default=[2.0, 1.4, 0.98, 0.686], 
                    help='不同尺度的深度估计损失权重')


def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    save_path = Path('checkpoints/scannet')
    save_path.makedirs_p()
    
    # 数据加载
    print("=> 加载scannet数据集 '{}'".format(args.data))
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    train_transform = Compose([
        RandomHorizontalFlip(),
        RandomScaleCrop(),
        ArrayToTensor(),
        normalize
    ])
    
    valid_transform = Compose([
        Scale(),
        ArrayToTensor(),
        normalize
    ])
    
    # 训练集
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        ttype=args.train_file,
        sequence_length=args.seq_length,
        sequence_gap=args.seq_gap,
        height=args.height,
        width=args.width,
    )
    
    # 验证集
    val_set = SequenceFolder(
        args.data,
        transform=valid_transform,
        seed=args.seed,
        ttype=args.val_file,
        sequence_length=args.seq_length,
        sequence_gap=args.seq_gap,
        height=args.height,
        width=args.width,
    )
    
    print('{} 个样本在训练集中, {} 个样本在验证集中'.format(len(train_set), len(val_set)))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # 创建模型
    print("=> 创建DELTAS模型")
    
    # 步骤1：兴趣点检测与描述 - SuperPoint
    config_sp = {
        'has_detector': True,
        'has_descriptor': True,
        'descriptor_dim': args.descriptor_dim,
        'top_k_keypoints': args.num_kps,
        'height': args.height,
        'width': args.width,
        'align_corners': args.align_corners,
        'detection_threshold': args.detection_threshold,
        'frac_superpoint': args.frac_superpoint,
        'nms_radius': args.nms_radius,
        'model_type': args.model_type,
    }
    
    supernet = superpoint.Superpoint(config_sp)
    
    # 步骤2：点匹配与三角测量 - TriangulationNet
    config_tri = {
        'depth_range': args.depth_range,
        'dist_ortogonal': args.dist_orthogonal,
        'kernel_size': args.kernel_size,
        'out_length': args.out_length,
        'has_confidence': args.do_confidence,
        'min_depth': args.mindepth,
        'max_depth': args.maxdepth,
        'align_corners': args.align_corners,
    }
    
    trinet = triangulation.TriangulationNet(config_tri)
    
    # 步骤3：稀疏点稠密化 - SparsetoDenseNet
    config_depth = {
        'min_depth': args.mindepth,
        'max_depth': args.maxdepth,
        'input_shape': (args.height, args.width, 1),
    }
    
    depthnet = densedepth.SparsetoDenseNet(config_depth)
    
    # 移动模型到GPU
    if torch.cuda.is_available():
        supernet = supernet.cuda()
        trinet = trinet.cuda()
        depthnet = depthnet.cuda()
        cudnn.benchmark = True
    
    # 定义优化器
    params = list(supernet.parameters()) + list(trinet.parameters()) + list(depthnet.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 可选：加载预训练模型或恢复训练
    start_epoch = 0
    if args.pretrained:
        print("=> 加载预训练模型 '{}'".format(args.pretrained))
        weights = torch.load(args.pretrained)
        supernet.load_state_dict(weights['state_dict'], strict=False)
        trinet.load_state_dict(weights['state_dict_tri'], strict=False)
        depthnet.load_state_dict(weights['state_dict_depth'], strict=False)
    
    if args.resume:
        print("=> 从检查点恢复训练 '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        supernet.load_state_dict(checkpoint['state_dict'])
        trinet.load_state_dict(checkpoint['state_dict_tri'])
        depthnet.load_state_dict(checkpoint['state_dict_depth'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 并行化模型
    if torch.cuda.device_count() > 1:
        supernet = torch.nn.DataParallel(supernet)
        trinet = torch.nn.DataParallel(trinet)
        depthnet = torch.nn.DataParallel(depthnet)
    
    # 训练循环
    best_error = float('inf')
    for epoch in range(start_epoch, args.epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch, args.lr)
        
        # 训练一个epoch
        train_epoch(train_loader, supernet, trinet, depthnet, optimizer, epoch)
        
        # 验证
        error = validate(val_loader, supernet, trinet, depthnet)
        
        # 保存检查点
        is_best = error < best_error
        best_error = min(error, best_error)
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': supernet.module.state_dict() if isinstance(supernet, torch.nn.DataParallel) else supernet.state_dict(),
                'state_dict_tri': trinet.module.state_dict() if isinstance(trinet, torch.nn.DataParallel) else trinet.state_dict(),
                'state_dict_depth': depthnet.module.state_dict() if isinstance(depthnet, torch.nn.DataParallel) else depthnet.state_dict(),
                'best_error': best_error,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=save_path/f'checkpoint_epoch_{epoch+1}.pth.tar')
    # 训练完成后的最终评估
    print("\n" + "="*60)
    print("训练完成！正在进行最终评估...")
    print("="*60)
    
    # 进行最终验证并获取详细指标
    final_errors = validate_with_detailed_metrics(val_loader, supernet, trinet, depthnet)
    
    # 输出所有评估指标
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    print("\n最终评估结果:")
    print("-" * 50)
    for i, name in enumerate(error_names):
        print(f"{name:>10}: {final_errors[i]:.6f}")
    print("-" * 50)
    print(f"最佳 abs_rel: {best_error:.6f}")
    print("="*60)

def validate_with_detailed_metrics(val_loader, supernet, trinet, depthnet):
    """验证模型并返回详细的评估指标"""
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    errors_depth = AverageMeter(i=len(error_names))
    
    # 切换到评估模式
    supernet.eval()
    trinet.eval()
    depthnet.eval()
    
    with torch.no_grad():
        for i, (tgt_img, ref_imgs, poses, intrinsics, tgt_depth, ref_depths) in enumerate(val_loader):
            # 将数据移动到GPU
            if torch.cuda.is_available():
                tgt_img = tgt_img.cuda()
                ref_imgs = [img.cuda() for img in ref_imgs]
                poses = [pose.cuda() for pose in poses]
                intrinsics = intrinsics.cuda()
                tgt_depth = tgt_depth.cuda()
                ref_depths = [depth.cuda() for depth in ref_depths]
            
            # 构建对称输入
            img_var = make_symmetric(tgt_img, ref_imgs)
            
            # 步骤1：兴趣点检测与描述
            data_sp = {'img': img_var, 'process_tsp': 'ts'}
            pred_sp = supernet(data_sp)
            
            batch_sz = tgt_img.shape[0]
            img_var = img_var[:batch_sz]
            
            # 姿态和内参
            seq_val = args.seq_length - 1
            pose = torch.cat(poses, 1)
            pose = pose_square(pose)
            
            # 深度
            depth = tgt_depth
            depth_ref = torch.stack(ref_depths, 1)
            
            # 关键点和描述子
            keypoints = pred_sp['keypoints'][:batch_sz]
            features = pred_sp['features'][:batch_sz]
            skip_half = pred_sp['skip_half'][:batch_sz]
            skip_quarter = pred_sp['skip_quarter'][:batch_sz]
            skip_eight = pred_sp['skip_eight'][:batch_sz]
            skip_sixteenth = pred_sp['skip_sixteenth'][:batch_sz]
            scores = pred_sp['scores'][:batch_sz]
            desc = pred_sp['descriptors']
            desc_anc = desc[:batch_sz, :, :, :]
            desc_view = desc[batch_sz:, :, :, :]
            desc_view = reorder_desc(desc_view, batch_sz)
            
            # 步骤2：点匹配与三角测量
            data_sd = {
                'iter': i,
                'intrinsics': intrinsics,
                'pose': pose,
                'depth': depth,
                'ref_depths': depth_ref,
                'scores': scores,
                'keypoints': keypoints,
                'descriptors': desc_anc,
                'descriptors_views': desc_view,
                'img_shape': tgt_img.shape,
                'sequence_length': seq_val
            }
            pred_sd = trinet(data_sd)
            
            range_mask_view = pred_sd['range_kp']
            range_mask = torch.sum(range_mask_view, 1)
            keypoints_3d = pred_sd['keypoints_3d']
            
            # 步骤3：稀疏点稠密化
            data_dd = {
                'anchor_keypoints': keypoints,
                'keypoints_3d': keypoints_3d,
                'sequence_length': args.seq_length,
                'skip_sixteenth': skip_sixteenth,
                'range_mask': range_mask,
                'features': features,
                'skip_half': skip_half,
                'skip_quarter': skip_quarter,
                'skip_eight': skip_eight
            }
            pred_dd = depthnet(data_dd)
            output = pred_dd['dense_depth']
            
            # 计算指标
            mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)
            mask.detach_()
            output = output.squeeze(1)
            errors_depth.update(compute_errors(tgt_depth, output, mask, False))
            
            if i % args.print_freq == 0:
                print('最终评估: [{0}/{1}]'.format(i, len(val_loader)))
    
    return errors_depth.avg

def train_epoch(train_loader, supernet, trinet, depthnet, optimizer, epoch):
    """训练一个epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ip = AverageMeter()  # 兴趣点检测损失
    losses_2d = AverageMeter()  # 2D匹配损失
    losses_3d = AverageMeter()  # 3D三角测量损失
    losses_sm = AverageMeter()  # 平滑损失
    losses_d = [AverageMeter() for _ in range(4)]  # 不同尺度的深度估计损失
    
    # 切换到训练模式
    supernet.train()
    trinet.train()
    depthnet.train()
    
    end = time.time()
    
    for i, (tgt_img, ref_imgs, poses, intrinsics, tgt_depth, ref_depths) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)
        
        # 将数据移动到GPU
        if torch.cuda.is_available():
            tgt_img = tgt_img.cuda()
            ref_imgs = [img.cuda() for img in ref_imgs]
            poses = [pose.cuda() for pose in poses]
            intrinsics = intrinsics.cuda()
            tgt_depth = tgt_depth.cuda()
            ref_depths = [depth.cuda() for depth in ref_depths]
        
        # 构建对称输入
        img_var = make_symmetric(tgt_img, ref_imgs)
        
        # 步骤1：兴趣点检测与描述
        data_sp = {'img': img_var, 'process_tsp': 'ts'}  # t是检测器，s是描述子
        pred_sp = supernet(data_sp)
        
        batch_sz = tgt_img.shape[0]
        img_var = img_var[:batch_sz]
        
        # 姿态和内参
        seq_val = args.seq_length - 1
        pose = torch.cat(poses, 1)
        pose = pose_square(pose)
        
        # 深度
        depth = tgt_depth
        depth_ref = torch.stack(ref_depths, 1)
        
        # 关键点和描述子
        keypoints = pred_sp['keypoints'][:batch_sz]
        features = pred_sp['features'][:batch_sz]
        skip_half = pred_sp['skip_half'][:batch_sz]
        skip_quarter = pred_sp['skip_quarter'][:batch_sz]
        skip_eight = pred_sp['skip_eight'][:batch_sz]
        skip_sixteenth = pred_sp['skip_sixteenth'][:batch_sz]
        scores = pred_sp['scores'][:batch_sz]
        desc = pred_sp['descriptors']
        desc_anc = desc[:batch_sz, :, :, :]
        desc_view = desc[batch_sz:, :, :, :]
        desc_view = reorder_desc(desc_view, batch_sz)
        
        # 步骤2：点匹配与三角测量
        data_sd = {
            'iter': i,
            'intrinsics': intrinsics,
            'pose': pose,
            'depth': depth,
            'ref_depths': depth_ref,
            'scores': scores,
            'keypoints': keypoints,
            'descriptors': desc_anc,
            'descriptors_views': desc_view,
            'img_shape': tgt_img.shape,
            'sequence_length': seq_val
        }
        pred_sd = trinet(data_sd)
        
        view_matches = pred_sd['multiview_matches']
        anchor_keypoints = pred_sd['keypoints']
        keypoints3d_gt = pred_sd['keypoints3d_gt']
        range_mask_view = pred_sd['range_kp']
        range_mask = torch.sum(range_mask_view, 1)
        
        keypoints_3d = pred_sd['keypoints_3d']
        kp3d_val = keypoints_3d[:, :, 2].view(-1, 1).t()
        kp3d_filter = (range_mask > 0).view(-1, 1).t()
        kp3d_filter = (kp3d_filter) & (kp3d_val > args.mindepth) & (kp3d_val < args.maxdepth)
        
        # 步骤3：稀疏点稠密化
        data_dd = {
            'anchor_keypoints': keypoints,
            'keypoints_3d': keypoints_3d,
            'sequence_length': args.seq_length,
            'skip_sixteenth': skip_sixteenth,
            'range_mask': range_mask,
            'features': features,
            'skip_half': skip_half,
            'skip_quarter': skip_quarter,
            'skip_eight': skip_eight
        }
        pred_dd = depthnet(data_dd)
        output = pred_dd['dense_depth']
        
        # 计算损失
        # 1. 兴趣点检测损失
        loss_ip = compute_interest_point_loss(pred_sp, tgt_depth)
        
        # 2. 2D匹配损失
        loss_2d = compute_2d_matching_loss(view_matches, anchor_keypoints, range_mask_view)
        
        # 3. 3D三角测量损失
        loss_3d = compute_3d_triangulation_loss(keypoints_3d, keypoints3d_gt, range_mask)
        
        # 4. 平滑损失
        loss_sm = compute_smoothness_loss(output, tgt_img)
        
        # 5. 多尺度深度估计损失
        mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)
        mask.detach_()
        
        loss_d = []
        # 检查是否有多尺度输出
        if 'multiscale' in pred_dd and pred_dd['multiscale'] is not None:
            # 使用多尺度输出
            for i, scale_output in enumerate(pred_dd['multiscale']):
                scale_loss = compute_depth_loss(scale_output, tgt_depth, mask)
                loss_d.append(scale_loss)
                if i < len(losses_d):  # 确保索引不越界
                    losses_d[i].update(scale_loss.item())
        else:
            # 如果没有多尺度输出，只使用主要的dense_depth输出
            scale_loss = compute_depth_loss(pred_dd['dense_depth'], tgt_depth, mask)
            loss_d.append(scale_loss)
            losses_d[0].update(scale_loss.item())
        
        # 总损失
        loss = args.w_ip * loss_ip + args.w_2d * loss_2d + args.w_3d * loss_3d + args.w_sm * loss_sm
        for i, ld in enumerate(loss_d):
            if i < len(args.w_d):  # 确保权重索引不越界
                loss += args.w_d[i] * ld
            else:
                loss += args.w_d[0] * ld  # 使用第一个权重作为默认值
        
        # 更新损失记录
        losses.update(loss.item())
        losses_ip.update(loss_ip.item())
        losses_2d.update(loss_2d.item())
        losses_3d.update(loss_3d.item())
        losses_sm.update(loss_sm.item())
        
        # 计算梯度并优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 测量批处理时间
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'IP {loss_ip.val:.4f} 2D {loss_2d.val:.4f} 3D {loss_3d.val:.4f} SM {loss_sm.val:.4f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_ip=losses_ip,
                   loss_2d=losses_2d, loss_3d=losses_3d, loss_sm=losses_sm))


def validate(val_loader, supernet, trinet, depthnet):
    """验证模型"""
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    errors_depth = AverageMeter(i=len(error_names))
    
    # 切换到评估模式
    supernet.eval()
    trinet.eval()
    depthnet.eval()
    
    with torch.no_grad():
        for i, (tgt_img, ref_imgs, poses, intrinsics, tgt_depth, ref_depths) in enumerate(val_loader):
            # 将数据移动到GPU
            if torch.cuda.is_available():
                tgt_img = tgt_img.cuda()
                ref_imgs = [img.cuda() for img in ref_imgs]
                poses = [pose.cuda() for pose in poses]
                intrinsics = intrinsics.cuda()
                tgt_depth = tgt_depth.cuda()
                ref_depths = [depth.cuda() for depth in ref_depths]
            
            # 构建对称输入
            img_var = make_symmetric(tgt_img, ref_imgs)
            
            # 步骤1：兴趣点检测与描述
            data_sp = {'img': img_var, 'process_tsp': 'ts'}
            pred_sp = supernet(data_sp)
            
            batch_sz = tgt_img.shape[0]
            img_var = img_var[:batch_sz]
            
            # 姿态和内参
            seq_val = args.seq_length - 1
            pose = torch.cat(poses, 1)
            pose = pose_square(pose)
            
            # 深度
            depth = tgt_depth
            depth_ref = torch.stack(ref_depths, 1)
            
            # 关键点和描述子
            keypoints = pred_sp['keypoints'][:batch_sz]
            features = pred_sp['features'][:batch_sz]
            skip_half = pred_sp['skip_half'][:batch_sz]
            skip_quarter = pred_sp['skip_quarter'][:batch_sz]
            skip_eight = pred_sp['skip_eight'][:batch_sz]
            skip_sixteenth = pred_sp['skip_sixteenth'][:batch_sz]
            scores = pred_sp['scores'][:batch_sz]
            desc = pred_sp['descriptors']
            desc_anc = desc[:batch_sz, :, :, :]
            desc_view = desc[batch_sz:, :, :, :]
            desc_view = reorder_desc(desc_view, batch_sz)
            
            # 步骤2：点匹配与三角测量
            data_sd = {
                'iter': i,
                'intrinsics': intrinsics,
                'pose': pose,
                'depth': depth,
                'ref_depths': depth_ref,
                'scores': scores,
                'keypoints': keypoints,
                'descriptors': desc_anc,
                'descriptors_views': desc_view,
                'img_shape': tgt_img.shape,
                'sequence_length': seq_val
            }
            pred_sd = trinet(data_sd)
            
            range_mask_view = pred_sd['range_kp']
            range_mask = torch.sum(range_mask_view, 1)
            keypoints_3d = pred_sd['keypoints_3d']
            
            # 步骤3：稀疏点稠密化
            data_dd = {
                'anchor_keypoints': keypoints,
                'keypoints_3d': keypoints_3d,
                'sequence_length': args.seq_length,
                'skip_sixteenth': skip_sixteenth,
                'range_mask': range_mask,
                'features': features,
                'skip_half': skip_half,
                'skip_quarter': skip_quarter,
                'skip_eight': skip_eight
            }
            pred_dd = depthnet(data_dd)
            output = pred_dd['dense_depth']
            
            # 计算指标
            mask = (tgt_depth <= args.maxdepth) & (tgt_depth >= args.mindepth) & (tgt_depth == tgt_depth)
            mask.detach_()
            output = output.squeeze(1)
            errors_depth.update(compute_errors(tgt_depth, output, mask, False))
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Depth Error {errors_depth.avg[1]:.4f} ({errors_depth.avg[0]:.4f})'.format(
                       i, len(val_loader), errors_depth=errors_depth))
    
    print('\n测试结果: Depth Error {:.4f} ({:.4f})'.format(errors_depth.avg[1], errors_depth.avg[0]))
    return errors_depth.avg[0]  # 返回abs_rel作为主要指标


def compute_interest_point_loss(pred_sp, gt_depth):
    """计算兴趣点检测损失"""
    # 使用scores代替heatmap
    if 'scores' in pred_sp:
        # 对scores进行softmax处理得到类似heatmap的概率分布
        scores = torch.nn.functional.softmax(pred_sp['scores'], 1)[:, :-1]  # 去掉最后一个通道（背景）
        b, c, h, w = scores.shape
        # 将多通道scores合并为单通道heatmap
        heatmap = torch.sum(scores, dim=1)  # [B, H, W]
    else:
        # 如果没有scores，返回零损失
        return torch.tensor(0.0, device=gt_depth.device, requires_grad=True)
    
    # 调整heatmap尺寸以匹配gt_depth
    if heatmap.shape[-2:] != gt_depth.shape[-2:]:
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(1), 
            size=gt_depth.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    
    valid_mask = (gt_depth > 0) & (gt_depth < args.maxdepth)
    valid_mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
    
    # 计算损失：在有效深度区域，heatmap应该有更高的响应
    loss = torch.mean(torch.abs(heatmap * valid_mask.float() - heatmap))
    return loss


def compute_2d_matching_loss(view_matches, anchor_keypoints, range_mask_view):
    """计算2D匹配损失"""
    # 计算匹配点之间的距离作为损失
    batch_size = view_matches.shape[0]
    device = view_matches.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for b in range(batch_size):
        for v in range(view_matches.shape[1]):
            mask = range_mask_view[b, v] > 0
            if mask.sum() > 0:
                matched_points = view_matches[b, v][mask]
                anchor_points = anchor_keypoints[b][mask]
                loss = loss + torch.mean(torch.norm(matched_points - anchor_points, dim=1))
    
    return loss / (batch_size + 1e-8)

def compute_3d_triangulation_loss(pred_points, gt_points, mask):
    """计算3D三角测量损失"""
    valid_mask = mask.unsqueeze(2).expand_as(pred_points) > 0
    diff = torch.abs(pred_points - gt_points)
    diff = diff * valid_mask.float()
    loss = diff.sum() / (valid_mask.sum() + 1e-8)
    return loss


def compute_smoothness_loss(depth, image):
    """计算平滑损失"""
    depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    
    image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
    image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)
    
    weights_x = torch.exp(-image_dx)
    weights_y = torch.exp(-image_dy)
    
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def compute_depth_loss(pred, target, mask):
    """计算深度估计损失"""
    pred = pred.squeeze(1)
    
    # 确保target和mask有正确的维度
    if len(target.shape) == 3:  # [B, H, W]
        target = target.unsqueeze(1)  # [B, 1, H, W]
    if len(mask.shape) == 3:  # [B, H, W]
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
    
    # 如果预测和目标尺寸不匹配，调整目标尺寸以匹配预测
    if pred.shape[-2:] != target.shape[-2:]:
        # 下采样目标深度图以匹配预测尺寸
        target_resized = torch.nn.functional.interpolate(
            target.float(), 
            size=pred.shape[-2:], 
            mode='nearest'
        )
        # 同样调整mask的尺寸
        mask_resized = torch.nn.functional.interpolate(
            mask.float(), 
            size=pred.shape[-2:], 
            mode='nearest'
        ).bool()
    else:
        target_resized = target
        mask_resized = mask
    
    # 移除添加的维度以匹配原始逻辑
    target_resized = target_resized.squeeze(1)
    mask_resized = mask_resized.squeeze(1)
    
    diff = torch.abs(pred - target_resized) * mask_resized.float()
    loss = torch.sum(diff) / (mask_resized.sum() + 1e-8)
    return loss


def reorder_desc(desc, batch_size):
    """重新排序描述子"""
    seq_len = desc.shape[0] // batch_size
    desc_reordered = []
    for i in range(seq_len):
        desc_reordered.append(desc[i::seq_len])
    return torch.stack(desc_reordered, 1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """保存检查点"""
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copy(filename, best_file)


def adjust_learning_rate(optimizer, epoch, lr):
    """调整学习率"""
    lr = lr * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class RandomScaleCrop(object):
    """随机缩放和裁剪"""
    def __call__(self, images, depths, intrinsics, height, width):
        out_h = height
        out_w = width
        
        scaled_images = []
        scaled_depths = []
        
        for idx, (img, depth) in enumerate(zip(images, depths)):
            h, w = img.shape[:2]
            
            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            
            img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
            
            # 调整内参
            if idx == 0:
                intrinsics[0, 0] *= (scaled_w / w)
                intrinsics[1, 1] *= (scaled_h / h)
                intrinsics[0, 2] *= (scaled_w / w)
                intrinsics[1, 2] *= (scaled_h / h)
            
            # 随机裁剪
            start_h = np.random.randint(0, scaled_h - out_h + 1)
            start_w = np.random.randint(0, scaled_w - out_w + 1)
            
            img = img[start_h:start_h+out_h, start_w:start_w+out_w]
            depth = depth[start_h:start_h+out_h, start_w:start_w+out_w]
            
            # 调整内参
            if idx == 0:
                intrinsics[0, 2] -= start_w
                intrinsics[1, 2] -= start_h
            
            scaled_images.append(img)
            scaled_depths.append(depth)
        
        return scaled_images, scaled_depths, intrinsics


class RandomHorizontalFlip(object):
    """随机水平翻转"""
    def __call__(self, images, depths, intrinsics, height, width):
        if np.random.random() < 0.5:
            flipped_images = [np.copy(np.fliplr(img)) for img in images]
            flipped_depths = [np.copy(np.fliplr(depth)) for depth in depths]
            w = images[0].shape[1]
            intrinsics[0, 2] = w - intrinsics[0, 2]
            return flipped_images, flipped_depths, intrinsics
        return images, depths, intrinsics


if __name__ == '__main__':
    main()