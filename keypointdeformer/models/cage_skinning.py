import pytorch3d.loss
import pytorch3d.utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from einops import rearrange

from ..utils.cages import deform_with_MVC
from ..utils.networks import Linear, MLPDeformer2, PointNetfeat
from ..utils.utils import normalize_to_box, sample_farthest_points


class CageSkinning(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--n_influence_ratio", type=float, help="", default=1.0)
        parser.add_argument("--lambda_init_points", type=float, help="", default=2.0)
        parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)
        parser.add_argument("--lambda_influence_predict_l2", type=float, help="", default=1e6)
        parser.add_argument("--iterations_init_points", type=float, help="", default=200)
        parser.add_argument("--no_optimize_cage", action="store_true", help="")
        parser.add_argument("--ico_sphere_div", type=int, help="", default=1)
        parser.add_argument("--n_fps", type=int, help="")
        return parser

    
    def __init__(self, opt):
        super(CageSkinning, self).__init__()
        
        self.opt = opt
        self.dim = self.opt.dim
        
        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices, template_faces)
        self.init_networks(opt.bottleneck_size, self.opt.dim, opt)
        self.init_optimizer()


    def create_cage(self):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(self.opt.ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.opt.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    

    def init_networks(self, bottleneck_size, dim, opt):
        # keypoint predictor
        shape_encoder_kpt = nn.Sequential(
            PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size),
            Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization))
        nd_decoder_kpt = MLPDeformer2(dim=dim, bottleneck_size=bottleneck_size, npoint=opt.n_keypoints,
                                residual=opt.d_residual, normalization=opt.normalization)
        self.keypoint_predictor = nn.Sequential(shape_encoder_kpt, nd_decoder_kpt)

        # influence predictor
        influence_size = self.opt.n_keypoints * self.template_vertices.shape[2]
        shape_encoder_influence = nn.Sequential(
            PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=influence_size),
            Linear(influence_size, influence_size, activation="lrelu", normalization=opt.normalization))
        dencoder_influence = nn.Sequential(
                Linear(influence_size, influence_size, activation="lrelu", normalization=opt.normalization),
                Linear(influence_size, influence_size, activation=None, normalization=None))
        self.influence_predictor = nn.Sequential(shape_encoder_influence, dencoder_influence)
        

    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.opt.n_keypoints, self.template_vertices.shape[2]), requires_grad=True)


    def init_optimizer(self):
        params = [{"params": self.influence_predictor.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.opt.lr)
        self.optimizer.add_param_group({'params': self.influence_param, 'lr': 10 * self.opt.lr})
        params = [{"params": self.keypoint_predictor.parameters()}]
        self.keypoint_optimizer = torch.optim.Adam(params, lr=self.opt.lr)


    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    
    def forward(self, source_shape, target_shape):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        B, _, _ = source_shape.shape

        self.target_shape = target_shape

        if target_shape is not None:
            shape = torch.cat([source_shape, target_shape], dim=0)
        else:
            shape = source_shape
        
        keypoints = self.keypoint_predictor(shape)
        keypoints = torch.clamp(keypoints, -1.0, 1.0)
        if target_shape is not None:
            source_keypoints, target_keypoints = torch.split(keypoints, B, dim=0)
        else:
            source_keypoints = keypoints

        self.shape = shape
        self.keypoints = keypoints
        
        n_fps = self.opt.n_fps if self.opt.n_fps else 2 * self.opt.n_keypoints
        self.init_keypoints = sample_farthest_points(shape, n_fps)

        if target_shape is not None:
            source_init_keypoints, target_init_keypoints = torch.split(self.init_keypoints, B, dim=0)
        else:
            source_init_keypoints = self.init_keypoints
            target_init_keypoints = None

        cage = self.template_vertices
        if not self.opt.no_optimize_cage:
            cage = self.optimize_cage(cage, source_shape)

        outputs = {
            "cage": cage.transpose(1, 2),
            "cage_face": self.template_faces,
            "source_keypoints": source_keypoints,
            "target_keypoints": target_keypoints,
            'source_init_keypoints': source_init_keypoints,
            'target_init_keypoints': target_init_keypoints
        }

        self.influence = self.influence_param[None]
        self.influence_offset = self.influence_predictor(source_shape)
        self.influence_offset = rearrange(
            self.influence_offset, 'b (k c) -> b k c', k=self.influence.shape[1], c=self.influence.shape[2])
        self.influence = self.influence + self.influence_offset
        
        distance = torch.sum((source_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1)
        n_influence = int((distance.shape[2] / distance.shape[1]) * self.opt.n_influence_ratio)
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence * keep

        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        deformed_shapes, weights, _ = deform_with_MVC(
            cage, new_cage, self.template_faces.expand(B, -1, -1), source_shape.transpose(1, 2), verbose=True)
        
        self.deformed_shapes = deformed_shapes
        
        outputs.update({
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": self.deformed_shapes,
            "weight": weights,
            "influence": influence})
        
        return outputs
            

    def compute_loss(self, iteration):
        losses = {}

        if self.opt.lambda_init_points > 0:
            init_points_loss = pytorch3d.loss.chamfer_distance(
                rearrange(self.keypoints, 'b d n -> b n d'), 
                rearrange(self.init_keypoints, 'b d n -> b n d'))[0]
            losses['init_points'] = self.opt.lambda_init_points * init_points_loss

        if self.opt.lambda_chamfer > 0:
            chamfer_loss = pytorch3d.loss.chamfer_distance(
                self.deformed_shapes, rearrange(self.target_shape, 'b d n -> b n d'))[0]
            losses['chamfer'] = self.opt.lambda_chamfer * chamfer_loss

        if self.opt.lambda_influence_predict_l2 > 0:
            losses['influence_predict_l2'] = self.opt.lambda_influence_predict_l2 * torch.mean(self.influence_offset ** 2)

        return losses

    
    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)
        

    def optimize(self, losses, iteration):
        self.keypoint_optimizer.zero_grad()
        self.optimizer.zero_grad()

        if iteration < self.opt.iterations_init_points:
            keypoints_loss = self._sum_losses(losses, ['init_points'])
            keypoints_loss.backward(retain_graph=True)
            self.keypoint_optimizer.step()

        if iteration >= self.opt.iterations_init_points:
            loss = self._sum_losses(losses, ['chamfer', 'influence_predict_l2', 'init_points'])
            loss.backward()
            self.optimizer.step()
            self.keypoint_optimizer.step()
