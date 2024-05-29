import torch
import cv2
from sklearn.cluster import KMeans
import numpy as np


REORDER_MTX = torch.tensor([
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0]
]).cuda().float()

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)    
    R[:, 1, 0] = 2 * (x * y + r * z)    
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)    
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def rotation_matrix(angle_x, angle_y, angle_z):
    # Convert angles to radians
    rad_x = torch.deg2rad(torch.tensor(angle_x))
    rad_y = torch.deg2rad(torch.tensor(angle_y))
    rad_z = torch.deg2rad(torch.tensor(angle_z))

    # Compute sine and cosine of the angles
    cos_x = torch.cos(rad_x)
    sin_x = torch.sin(rad_x)
    cos_y = torch.cos(rad_y)
    sin_y = torch.sin(rad_y)
    cos_z = torch.cos(rad_z)
    sin_z = torch.sin(rad_z)

    # Construct the rotation matrix
    Rx = torch.tensor([[1, 0, 0],
                   [0, cos_x, -sin_x],
                   [0, sin_x, cos_x]])

    Ry = torch.tensor([[cos_y, 0, sin_y],
                   [0, 1, 0],
                   [-sin_y, 0, cos_y]])

    Rz = torch.tensor([[cos_z, -sin_z, 0],
                   [sin_z, cos_z, 0],
                   [0, 0, 1]])

    # Combine the rotation matrices
    rotation_matrix = Rz @ Ry @ Rx

    return rotation_matrix

def prune(system, idx=0, p1=3.0, kmeans_t=0.3):
    system.sam_clip_ae.eval()
    ooi_masks = getattr(system.geometry, f"ooi_masks_{idx}").cpu().numpy()
    ooi_masks = cv2.blur(ooi_masks.astype(np.float32), (5, 5)) > 0
    ooi_mask = torch.tensor(ooi_masks.reshape(-1).astype(np.uint8), device='cuda').float()
    idx = torch.arange(len(ooi_mask), device='cuda')[ooi_mask.bool()]
    
    lang_feat = system.geometry.get_language_feature[ooi_mask.bool()]
    lang_feat = lang_feat / (lang_feat.norm(2, dim=-1, keepdim=True) + 1e-3)
    original_ooi_mask = ooi_mask.clone()

    # filter with color by KMeans
    kmeans = KMeans(n_init='auto', n_clusters=10)
    kmeans.fit(lang_feat.detach().cpu())
    labels = kmeans.labels_
    _ = [(labels==i).sum() for i in np.unique(labels)]
    max_label = _.index(max(_))
    dist = ((kmeans.cluster_centers_ - kmeans.cluster_centers_[max_label:max_label+1]) **2).sum(-1)**.5
    for label, num in enumerate(_):
        if dist[label] > kmeans_t:
            ooi_mask[idx[labels == label]] = False
            system.geometry._delete_mask[idx[labels == label]] = 0.
        
    p = p1
    # filter Gaussians 
    mean, std = lang_feat.mean(0), lang_feat.std(0)
    outlier = torch.logical_or(lang_feat < mean -  p * std, lang_feat > mean + p * std).sum(-1) > 0
    ooi_mask[idx[outlier]] = False
    system.geometry._delete_mask[idx[outlier]] = 0.
