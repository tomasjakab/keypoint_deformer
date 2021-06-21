import time

import numpy as np
import pytorch3d.io
import torch
from einops import repeat


def sample_farthest_points(points, num_samples, return_index=False):
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)
    
    index = torch.randint(n, [b], device=points.device)
    
    gather_index = repeat(index, 'b -> b c 1', c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, 'b -> b c 1', c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1))

    if return_index:
        return sampled, indexes
    else:
        return sampled


def resample_mesh(mesh, n_points):
    points, normals = pytorch3d.ops.sample_points_from_meshes(mesh, n_points, return_normals=True)
    points = torch.cat([points[0], normals[0]], dim=-1)
    return points


def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()
    
    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        input = input - centroid
        in_shape = list(input.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(input).view(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance



class Timer(object):
  def __init__(self, name=None, acc=False, avg=False):
    self.name = name
    self.acc = acc
    self.avg = avg
    self.total = 0.0
    self.iters = 0
    self.tstart = time.time()

  def __enter__(self):
    self.start()

  def __exit__(self, type, value, traceback):
    self.stop()

  def start(self):
    self.tstart = time.time()

  def stop(self):
    self.iters += 1
    self.total += time.time() - self.tstart
    if not self.acc:
      self.reset()

  def reset(self):
    name_string = ''
    if self.name:
      name_string = '[' + self.name + '] '
    value = self.total
    msg = 'Elapsed'
    if self.avg:
      value /= self.iters
      msg = 'Avg elapsed'
    print('%s%s: %.4f' % (name_string, msg, value))
    if not self.avg:
        self.total = 0.0
