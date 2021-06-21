import json
import os
import time
from datetime import datetime

import numpy as np
import pytorch3d.io
import torch
import torch.nn.parallel
import torch.utils.data
from keypointdeformer.datasets import get_dataset
from keypointdeformer.models import get_model
from keypointdeformer.options.base_options import BaseOptions
from keypointdeformer.utils import io
from keypointdeformer.utils.cages import deform_with_MVC
from keypointdeformer.utils.nn import load_network, save_network, weights_init
from keypointdeformer.utils.utils import Timer
from tensorboardX import SummaryWriter

CHECKPOINTS_DIR = 'checkpoints'
CHECKPOINT_EXT = '.pth'


def write_losses(writer, losses, step):
    for name, value in losses.items():
        writer.add_scalar('loss/' + name, value, global_step=step)


def save_normalization(file_path, center, scale):
    with open(file_path, 'w') as f:
        json.dump({'center': [str(x) for x in center.cpu().numpy()], 'scale': str(scale.cpu().numpy()[0])}, f)


def save_data_keypoints(data, save_dir, name):
    if name in data:
        io.save_keypoints(os.path.join(save_dir, name + '.txt'), data[name])


def save_data_txt(f, data, fmt):
    np.savetxt(f, data.cpu().detach().numpy(), fmt=fmt)


def save_pts(f, points, normals=None):
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    io.save_pts(f, points.cpu().detach().numpy(), normals=normals)


def save_ply(f, verts, faces):
    pytorch3d.io.save_ply(f, verts.cpu(), faces=faces.cpu())

    
def save_output(save_dir_root, data, outputs, save_mesh=True, save_auxilary=True):
    name = data['source_file']
    save_dir = os.path.join(save_dir_root, name)
    os.makedirs(save_dir, exist_ok=True)

    # save meshes
    if save_mesh and 'source_mesh' in data:
        io.save_mesh(os.path.join(save_dir, 'source_mesh.obj'), data["source_mesh"], data["source_face"])

        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'source_vertices.txt'), data["source_mesh"], '%0.6f')
            save_data_txt(os.path.join(save_dir, 'source_faces.txt'), data["source_face"], '%d')

        io.save_mesh(os.path.join(save_dir, 'target_mesh.obj'), data["target_mesh"], data["target_face"])

        if outputs is not None:
            deformed, weights, _ = deform_with_MVC(
                outputs["cage"][None], outputs["new_cage"][None], outputs["cage_face"][None], 
                data["source_mesh"][None], verbose=True)
            io.save_mesh(os.path.join(save_dir, 'deformed_mesh.obj'), deformed[0], data["source_face"])
            if save_auxilary:
                save_data_txt(os.path.join(save_dir, 'weights.txt'), weights[0], '%0.6f')

    # save pointclouds
    save_pts(os.path.join(save_dir, 'source_pointcloud.pts'), data['source_shape'], normals=data['source_normals'])
    if outputs is not None:
        save_pts(os.path.join(save_dir, 'deformed_pointcloud.pts'), outputs['deformed'])
        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'influence.txt'), outputs['influence'], '%0.6f')

    save_pts(os.path.join(save_dir, 'target_pointcloud.pts'), data['target_shape'], normals=data['target_normals'])
    
    # save cages
    if outputs is not None:
        save_ply(os.path.join(save_dir, 'cage.ply'), outputs["cage"], outputs["cage_face"])
        if save_auxilary:
            save_data_txt(os.path.join(save_dir, 'cage.txt'), outputs["cage"], '%0.6f')
        save_ply(os.path.join(save_dir, 'deformed_cage.ply'), outputs["new_cage"], outputs["cage_face"])
    
        if outputs is not None:
            io.save_keypoints(os.path.join(save_dir, 'source_keypoints.txt'), outputs["source_keypoints"].transpose(0, 1))
            io.save_keypoints(os.path.join(save_dir, 'target_keypoints.txt'), outputs["target_keypoints"].transpose(0, 1))
        
        save_data_keypoints(data, save_dir, 'source_keypoints_gt')
        save_data_keypoints(data, save_dir, 'target_keypoints_gt')

        io.save_keypoints(os.path.join(save_dir, 'source_init_keypoints.txt'), outputs['source_init_keypoints'].transpose(0, 1))
        io.save_keypoints(os.path.join(save_dir, 'target_init_keypoints.txt'), outputs['target_init_keypoints'].transpose(0, 1))
        
        if 'source_keypoints_gt_center' in data:
            save_normalization(os.path.join(save_dir, 'source_keypoints_gt_normalization.txt'), data['source_keypoints_gt_center'], data['source_keypoints_gt_scale'])

        if 'source_seg_points' in data:
            io.save_labelled_pointcloud(os.path.join(save_dir, 'source_seg_points.xyzrgb'), data['source_seg_points'].detach().cpu().numpy(), data['source_seg_labels'].detach().cpu().numpy())
            save_data_txt(os.path.join(save_dir, 'source_seg_labels.txt'), data['source_seg_labels'], '%d')


def split_batch(data, b, singleton_keys=[]):
    return {k: v[b] if k not in singleton_keys else v[0] for k, v in data.items()}


def save_outputs(outputs_save_dir, data, outputs, save_mesh=True):
    for b in range(data['source_shape'].shape[0]):
        save_output(
            outputs_save_dir, split_batch(data, b, singleton_keys=['cage_face']), 
            split_batch(outputs, b, singleton_keys=['cage_face']), save_mesh=save_mesh)


def get_data(dataset, data):
    data = dataset.uncollate(data)

    source_shape, target_shape = data["source_shape"], data["target_shape"]

    source_shape_t = source_shape.transpose(1, 2)
    target_shape_t = target_shape.transpose(1, 2)
    
    return source_shape_t, target_shape_t


def test(opt, save_subdir="test"):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    opt.phase = "test"
    dataset = get_dataset(opt.dataset)(opt)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
        collate_fn=dataset.collate,
        num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    net = get_model(opt.model)(opt).cuda()
    ckpt = opt.ckpt
    if not ckpt.startswith(os.path.sep):
        ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
    load_network(net, ckpt)
    
    net.eval()

    test_output_dir = os.path.join(log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    timer = Timer('step')
    with torch.no_grad():
        for data in dataloader:
            timer.stop()
            timer.start()
            
            # data
            data = dataset.uncollate(data)
        
            source_shape_t, target_shape_t = get_data(dataset, data)
            outputs = net(source_shape_t, target_shape=target_shape_t)

            save_outputs(os.path.join(log_dir, save_subdir), data, outputs)


def train(opt):
    log_dir = os.path.join(opt.log_dir, opt.name)
    checkpoints_dir = os.path.join(log_dir, CHECKPOINTS_DIR)

    dataset = get_dataset(opt.dataset)(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate,
        num_workers=opt.n_workers, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # network
    net = get_model(opt.model)(opt).cuda()
    net.apply(weights_init)
    if opt.ckpt:
        ckpt = opt.ckpt
        if not ckpt.startswith(os.path.sep):
            ckpt = os.path.join(checkpoints_dir, ckpt + CHECKPOINT_EXT)
        load_network(net, ckpt)

    # train
    net.train()
    t = 0

    # train
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = open(os.path.join(checkpoints_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")
    summary_dir = datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(logdir=os.path.join(checkpoints_dir, 'logs', summary_dir), flush_secs=5)

    if opt.iteration:
        t = opt.iteration
    
    iter_time_start = time.time()

    while t <= opt.n_iterations:
        for _, data in enumerate(dataloader):
            if t > opt.n_iterations:
                break

            source_shape_t, target_shape_t = get_data(dataset, data)
            outputs = net(source_shape_t, target_shape=target_shape_t)
            current_loss = net.compute_loss(t)
            net.optimize(current_loss, t)

            if t % opt.save_interval == 0:
                outputs_save_dir = os.path.join(checkpoints_dir, 'outputs', '%07d' % t)
                save_outputs(outputs_save_dir, data, outputs, save_mesh=False)
                save_network(net, checkpoints_dir, network_label="net", epoch_label=t)

            iter_time = time.time() - iter_time_start
            iter_time_start = time.time()
            if (t % opt.log_interval == 0):
                log_str = ''
                samples_sec = opt.batch_size / iter_time
                losses_str = ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()])
                log_str = "{:d}: iter {:.1f} sec, {:.1f} samples/sec {}".format(
                    t, iter_time, samples_sec, losses_str)

                print(log_str)
                log_file.write(log_str + "\n")

                write_losses(writer, current_loss, t)
            t += 1

    log_file.close()
    save_network(net, checkpoints_dir, network_label="net", epoch_label="final")


if __name__ == "__main__":
    parser = BaseOptions()
    opt = parser.parse()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if opt.phase == "test":
        test(opt, save_subdir=opt.subdir)
    elif opt.phase == "train":
        train(opt)
    else:
        raise ValueError()
