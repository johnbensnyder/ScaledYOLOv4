import math
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from utils.torch_utils import init_seeds, is_parallel

class YOLOBCEclsLoss(_Loss):
    
    def __init__(self, weight=1.0, pos_weight=1.0, device='cuda', reduction='mean'):
        super().__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.device = device
        self.weight = weight
        self.BCEcls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight])).to(self.device)
        
    def forward(self, input, target):
        assert len(input)==len(target)
        s = 3 / len(input)
        lcls = torch.zeros(1, device=self.device)
        for x, y in zip(input, target):
            lcls += self.BCEcls(x, y)
        lcls *= self.weight * s
        return lcls

class YOLOBCEobjLoss(_Loss):
    
    def __init__(self, balance, weight=1.0, pos_weight=1.0, device='cuda', reduction='mean'):
        super().__init__(reduction=reduction)
        self.balance = balance
        self.pos_weight = pos_weight
        self.device = device
        self.weight = weight
        self.BCEobj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight])).to(self.device)
        
    def forward(self, input, target):
        assert len(input)==len(target)==len(self.balance)
        s = 3 / len(self.balance) * (1.4 if len(self.balance) >= 4 else 1.)
        lobj = torch.zeros(1, device=self.device)
        for x, y, w in zip(input, target, self.balance):
            lobj += self.BCEobj(x, y) * w
        lobj *= self.weight * s
        return lobj

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

class YOLOGIoULoss(_Loss):
    
    def __init__(self, weight = 1.0, device='cuda'):
        super().__init__()
        self.weight = weight
        self.device = device
        
    def forward(self, box_ious):
        np = len(box_ious)
        s = 3 / np
        lbox = torch.zeros(1, device=self.device)
        for iou in box_ious:
            lbox += (1.0 - iou).mean() 
        lbox *= self.weight * s
        return lbox
    
def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3]), gi.clamp_(0, gain[2])))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
    
class YOLOLoss(_Loss):
    
    def __init__(self, model, balance=[4.0, 1.0, 0.4]):
        super().__init__()
        self.device = next(model.parameters()).device
        self.balance = balance
        self.model = model
        self.cp, self.cn = smooth_BCE(eps=0.0)
        self.cls_loss = YOLOBCEclsLoss(weight=model.cfg['cls'], 
                                       pos_weight=model.cfg['cls_pw'], 
                                       device=self.device, 
                                       reduction='mean')
        self.obj_loss = YOLOBCEobjLoss(self.balance, 
                                       weight=model.cfg['obj'], 
                                       pos_weight=model.cfg['obj_pw'], 
                                       device=self.device, 
                                       reduction='mean')
        self.GIoU_loss = YOLOGIoULoss(weight=model.cfg['giou'],
                                      device=self.device)
        
    def forward(self, inputs, targets):
        class_inputs = []
        obj_inputs = []
        box_ious = []
        class_targets = []
        obj_targets = []
        tcls, tbox, indices, anchors = build_targets(inputs, targets, self.model)
        nt = 0
        for i, pi in enumerate(inputs):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=self.device)
            n = b.shape[0]
            if n:
                nt += n
                ps = pi[b, a, gj, gi]
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(self.device)  # predicted box
                giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
                box_ious.append(giou)
                tobj[b, a, gj, gi] = (1.0 - self.model.gr) + self.model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio    
                obj_inputs.append(pi[..., 4])
                obj_targets.append(tobj)
                if self.model.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    class_inputs.append(ps[:, 5:])
                    class_targets.append(t)
        lbox = self.GIoU_loss(box_ious)
        lobj = self.obj_loss(obj_inputs, obj_targets)
        lcls = self.cls_loss(class_inputs, class_targets) if self.model.nc > 1 else torch.zeros(1, device=self.device)
        bs = tobj.shape[0]
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()