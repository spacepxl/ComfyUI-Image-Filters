import os
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def load_raft():
    model_dir = os.path.join(os.path.split(__file__)[0], "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_path = os.path.join(model_dir, str(raft_weights) + ".pth")
    
    if os.path.exists(raft_path):
        model = raft_large()
        model.load_state_dict(torch.load(raft_path))
    else:
        model = raft_large(weights=raft_weights, progress=True)
        torch.save(model.state_dict(), raft_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return (model, device)

def raft_flow(model, device, batch1, batch2):
    orig_H = batch1.shape[2]
    orig_W = batch1.shape[3]
    scale_factor = max(orig_H, orig_W) / 512
    new_H = int(((orig_H / scale_factor) // 8) * 8)
    new_W = int(((orig_W / scale_factor) // 8) * 8)
    
    if scale_factor > 1 or max(orig_H % 8, orig_W % 8) > 0:
        batch1_scaled = F.interpolate(batch1, size=(new_H, new_W), mode='bilinear')
        batch2_scaled = F.interpolate(batch2, size=(new_H, new_W), mode='bilinear')
        
        with torch.no_grad():
            flow = model(batch1_scaled.to(device), batch2_scaled.to(device))[-1]
        flow = F.interpolate(flow, size=(orig_H, orig_W), mode='bilinear')
        flow[:,0,:,:] *= orig_W / new_W
        flow[:,1,:,:] *= orig_H / new_H
    else:
        with torch.no_grad():
            flow = model(batch1.to(device), batch2.to(device))[-1]
    
    return flow.to(batch1.device)

def flow_warp(image, flow):
    B, C, H, W = image.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    grid = grid.to(image.device)
    vgrid = grid + flow
    
    # scale grid to [-1,1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(image, vgrid, mode='bicubic', padding_mode='border', align_corners=True)
    return output