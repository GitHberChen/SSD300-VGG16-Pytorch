# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0) # batch size
        # priors: [[cx,cy,w,h]]
        # loc_data.size(1),num_priors，训练时priors[:loc_data.size(1), :]=prior？
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            # .data是从Variable获取底层Tensor的主要方式。合并后，调用y = x.data仍然具有类似的语义。
            # 因此y是一个与x共享相同数据的Tensor ，它与x的计算历史无关，并且requires_grad=False。
            #
            # 但是，.data在某些情况下可能不太稳定。x.data的任何变化都不会被autograd跟踪，
            # 并且如果在反向传递中需要x，计算出的梯度会出错。一种更安全的替代方法是使用x.detach()，
            # 它也返回一个与requires_grad=False共享数据的Tensor，但是如果x需要反向传递，
            # 则它将使用autograd就地更改记录。
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        # loc_t [batch_size, prior_num, 4] 4===>center offset
        # conf_t [batch_size, prior_num]
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        # 目标所在的相对锚框偏移量以及置信度
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # one-hot 编码, pos表示不为背景的锚框
        pos = conf_t > 0
        # shape[batch_size, prior_num]===>[batch_size, match_num]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # 在最后一个维度上增加一个维度，并且拓展成loc_data的形状i.e.[1,1,1,1]or[0,0,0,0]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 取出其中非背景的部分
        loc_p = loc_data[pos_idx].view(-1, 4)  # [batch_size*match_num, 4]
        loc_t = loc_t[pos_idx].view(-1, 4)  # [batch_size*match_num, 4]
        # (1)如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss
        # (2)如果 reduce = True，那么 loss 返回的是标量
        # a)如果 size_average = True，返回 loss.mean();
        # b)如果 size_average = False，返回 loss.sum();
        # loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=True)#)reduction='sum')
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)  # shape[batch_size*prior_num, num_classes]
        # conf_t shape [batch_size, 1] 1===>truth_tag
        # batch_conf.gather===>与真实类别对于的置信度
        # log_sum_exp ===> log(sum(exp(x - x_max), dim=1, keepdim=True)) + x_max
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(dim=1, index=conf_t.view(-1, 1))

        # Hard Negative Mining
        # 原本：loss_c[pos] = 0
        loss_c[pos.view(-1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=True)#reduction='sum')
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
