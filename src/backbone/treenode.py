import torch
import torch.nn as nn
from src.cell_utils import list_sum
from .operations import blocks_keys, blocks_dict
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from backbone.binarized_cell import BinarizedCellFunction


class TreeNode(nn.Module):
    def __init__(self, in_channels, out_channels, stride, arch_parameters, downsample=None,
                 split_type='copy', merge_type='add', use_avg=True, bn_before_add=False,
                 path_drop_rate=0, use_zero_drop=True, drop_only_add=False, cell_drop_rate=0):
        super(TreeNode, self).__init__()
        self.arch_parameters = arch_parameters
        self.acc_edge = 0

        self.child_ops = nn.ModuleList()
        self.root_ops = nn.ModuleList()
        self.initlize_ops(self.root_ops, in_channels, out_channels, stride)
        self.initlize_ops(self.child_ops, out_channels, out_channels, stride=1)

        self.branch_num = 2
        self.depth = 3
        self.stride = stride
        if stride != 1:
            if downsample is None:
                self.downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
            else:
                self.downsample = downsample

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.merge_type = merge_type
        if self.merge_type == 'add':
            self.split_type = 'copy'

        self.use_avg = use_avg

        self.bn_before_add = bn_before_add
        self.path_drop_rate = path_drop_rate
        self.use_zero_drop = use_zero_drop
        self.drop_only_add = drop_only_add
        self.cell_drop_rate = cell_drop_rate

        self.branch_bns = None

        self.leave_nodes = 4
        self.internal_nodes = 3
        # do not use batchnorm before add
        """
        if self.bn_before_add and self.merge_type == 'add':
            branch_bns = []
            for _i in range(self.child_num):
                branch_bns.append(nn.BatchNorm2d(self.out_dim_list[_i]))
            self.branch_bns = nn.ModuleList(branch_bns)
        """

    def initlize_ops(self, ops_module_list, in_channel, out_channel, stride):
        for key in blocks_keys:
            op = blocks_dict[key](in_channel, out_channel, stride)
            ops_module_list.append(op)

    @property
    def child_num(self):
        # this make sure we inputs edges that we needed.
        # binarized outside cell function
        return (self.branch_num ** (self.depth+1) - 2) // 2

    def binarization(self):
        binary_dict = {}
        alpha = F.softmax(self.arch_parameters[self.acc_edge], dim=0)
        m = Categorical(alpha)
        idx = m.sample().cpu().item()
        #print(idx)
        self.acc_edge += 1
        binary_dict['alpha'] = alpha
        binary_dict['idx'] = idx
        return binary_dict

    def path_forward(self, root, curr_depth, output_list):

        if curr_depth == 0:
            self.acc_edge=0
            _ops = self.root_ops
        else:
            _ops = self.child_ops

        if self.idx is None:
            binary_left = self.binarization()
            binary_right = self.binarization()

            _left_ops = _ops[binary_left['idx']]
            _right_ops = _ops[binary_right['idx']]

            _left_one_tensors = BinarizedCellFunction.apply(binary_left['alpha'], root, binary_left['idx'])
            _right_one_tensors = BinarizedCellFunction.apply(binary_right['alpha'], root, binary_right['idx'])

            root_left = _left_ops(root) * _left_one_tensors
            root_right = _right_ops(root) * _right_one_tensors
        else:
            print(self.idx[0])
            print(_ops[self.idx[0]])
            _left_ops = _ops[self.idx[0]]
            self.idx.pop(0)
            _right_ops = _ops[self.idx[0]]
            self.idx.pop(0)

            root_left = _left_ops(root)
            root_right = _right_ops(root)

        curr_depth = curr_depth + 1

        if curr_depth == self.depth:
            output_list.append(root_left)
            output_list.append(root_right)
        else:
            self.path_forward(root_left, curr_depth, output_list)
            self.path_forward(root_right, curr_depth, output_list)

    def forward(self, x):
        if isinstance(x, list):
            x, self.idx = x[0], x[1]
        else:
            self.idx = None
        _shortcut = x
        if self.stride != 1:
            _shortcut = self.downsample(_shortcut)
        curr_depth = 0
        ## default split_type = 'copy'
        if self.split_type == 'copy':
            child_inputs = [x] * self.branch_num
        else:
            raise NotImplementedError

        child_outputs = []
        self.path_forward(root=x, curr_depth=curr_depth, output_list=child_outputs)
        child_outputs.append(_shortcut)
        if self.merge_type == 'concat':
            output = torch.cat(child_outputs, dim=1)

        if self.merge_type == 'add':
            output = list_sum(child_outputs)
            if self.use_avg:
                output /= self.child_num

        return output