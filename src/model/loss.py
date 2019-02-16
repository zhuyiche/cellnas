import torch
import torch.nn as nn


class CellDetLoss(nn.Module):
    def __init__(self):
        super(CellDetLoss, self).__init__()

    def forward(self, y_true, y_pred):
        #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        #y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        #print('y_true.shape: ', y_true.shape)
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        y_pred = y_pred[:, 1, :, :]
        #y_true = y_true[:, , :, :]
        #print('y_pred: ', y_pred)
        #print('y_true: ', y_true)
        #fkg_smooth = torch.pow((1 - y_pred), fkg_focal_smoother)
        #bkg_smooth = torch.pow(y_pred, bkg_focal_smoother)
        y_pred = y_pred.float()
        y_true = y_true.float()
        #print(0.1 * y_true * torch.log(y_pred))
        #print(0.9 * (1 - y_true) * torch.log(1 - y_pred))
        result = -torch.mean(0.1 * y_true * torch.log(y_pred) +
                             0.9 * (1 - y_true) * torch.log(1 - y_pred))
        # bkg_smooth * (1 - y_true) * torch.log(1 - y_pred))
        # fkg_smooth * y_true * torch.log(y_pred) +
        return result
