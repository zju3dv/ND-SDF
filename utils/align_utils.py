
# by dawnzyt, get any high resolution normal or depth map using pre-trained model
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
import os

def save_depth_output(img, depth, output_dir, img_name):
    # img: (h,w,3), np.array, rgb
    # depth: (h,w), np.array, depth∈[0,1]
    norm_depth = ((depth - depth.min()) / (depth.max() - depth.min())*255).astype(np.uint8)
    vis_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
    cat = np.concatenate([img, cv2.cvtColor(vis_depth, cv2.COLOR_BGR2RGB)], axis = 1)
    Image.fromarray(cat).save(os.path.join(output_dir, 'vis','{}.png'.format(img_name)))
    np.save(os.path.join(output_dir, 'res','{}.npy'.format(img_name)), depth)
def save_normal_output(img, normal, output_dir, img_name):
    # img: (h,w,3), np.array, rgb
    # normal: (h,w,3), np.array, normal∈[-1,1]
    normal = (normal + 1) / 2
    cat = np.concatenate([img, (normal * 255).astype(np.uint8)], axis = 1)
    Image.fromarray(cat).save(os.path.join(output_dir, 'vis','{}.png'.format(img_name)))
    np.save(os.path.join(output_dir, 'res','{}.npy'.format(img_name)), normal)

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
            
# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only 
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    return R

# align depth2 to depth1 in the y direction
# [s1,e1), [s2,e2) are the overlapping part of depth1 and depth2
def align_y(depth1, depth2, s1, e1, s2, e2):
    # depth1, depth2: (1,h,w), tensor
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[2] == depth2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, s2:e2, :], depth1[:, s1:e1, :], torch.ones_like(depth1[:, s1:e1, :]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1] + depth2.shape[1] - (e1 - s1), depth1.shape[2]))

    if s1>0: # depth1 is above depth2, 即从下到上对齐
        result[:, :s1, :] = depth1[:, :s1, :]
        result[:, depth1.shape[1]:, :] = depth2_aligned[:, e2:, :]

        weight = np.linspace(1, 0, (e1-s1))[None, :, None]
        result[:, s1:depth1.shape[1], :] = depth1[:, s1:, :] * weight + depth2_aligned[:, :e2, :] * (1 - weight)
    elif s1==0: # depth1 is below depth2, 即从上到下对齐
        result[:, depth2.shape[1]:, :] = depth1[:, e1:, :]
        result[:, :s2, :] = depth2_aligned[:, :s2, :]
        weight = np.linspace(1, 0, (e1-s1))[None, :, None]
        result[:, s2:e2, :] = depth2_aligned[:, s2:e2, :] * weight + depth1[:, s1:e1, :] * (1 - weight)
    return result

# align normal2 to normal1 in the y direction
# [s1,e1), [s2,e2) are the overlapping part of normal1 and normal2
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    # normal1, normal2: (3,h,w), np.array
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2]))

    if s1>0: # normal1 is above normal2, 即从下到上对齐
        result[:, :s1, :] = normal1[:, :s1, :]
        result[:, normal1.shape[1]:, :] = normal2_aligned[:, e2:, :]

        weight = np.linspace(1, 0, (e1-s1))[None, :, None]
        
        result[:, s1:normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[:, :e2, :] * (1 - weight)
        result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    elif s1==0: # normal1 is below normal2, 即从上到下对齐
        result[:, normal2.shape[1]:, :] = normal1[:, e1:, :]
        result[:, :s2, :] = normal2_aligned[:, :s2, :]
        weight = np.linspace(1, 0, (e1-s1))[None, :, None]
        result[:, s2:e2, :] = normal2_aligned[:, s2:e2, :] * weight + normal1[:, s1:e1, :] * (1 - weight)
        result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)):int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

# align depth and normal maps
# aimed to support any raw_size of the prior model, and support any high resolution input image.
# eg: omnidata, raw size:384×384, input image size: (1080,1920)
class Aligner():
    def __init__(self, raw_size=(384,384)) -> None:
        # raw_size: (rh,rw)
        self.raw_size = raw_size # (rh,rw)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # transform fn, input: np.array(h,w,3), output: tensor(3,rh,rw)
        # default: input is rgb and uint8
        self.default_depth_trans_fn = transforms.Compose([ 
            transforms.ToPILImage(),
            transforms.Resize(raw_size, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.default_normal_trans_fn = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(raw_size, Image.BILINEAR),
            transforms.ToTensor(),
        ])

    def get_overlapping(self, h, w, rh, rw):
        oh, ow = int(rh*2/3), int(rw*2/3) # overlapping part of direction y and x
        ows = [0] # ows[i]: i th and i-1 th column overlapping part
        ohs = [0]
        ofs_xs = []
        ofs_ys = []
        for x in range(0,w,rw-ow):
            ofs_x = x
            if x+rw>w:
                ofs_x = w-rw
            ofs_xs.append(ofs_x)
            if ofs_x + rw == w: # last
                break
        for y in range(0,h,rh-oh):
            ofs_y = y
            if y+rh>h:
                ofs_y = h-rh
            ofs_ys.append(ofs_y)
            if ofs_y + rh == h:
                break
        for i in range(1, len(ofs_xs)):
            ows.append(rw-(ofs_xs[i]-ofs_xs[i-1]))
        for i in range(1, len(ofs_ys)):
            ohs.append(rh-(ofs_ys[i]-ofs_ys[i-1]))
        return ows, ohs, ofs_xs, ofs_ys
    def get_highres_normal(self, img, normal_fn, trans_fn=None, downscale=1.0, mode='TBLR'):
        # img: (h,w,3), np.array, rgb
        # normal_fn: input: (b,3,rh,rw), output: (b,3,rh,rw), normal_fn output∈[0,1]
        # trans_fn: input: np.array(h,w,3), output: tensor(3,rh,rw), prepare for the prior model
        # mode: 'MM' or 'TBLR', 'MM' means align from middle to both sides, 'TBLR' means align from top to bottom and left to right
        if downscale!=1.0:
            origin_h, origin_w = img.shape[:2]
            img = cv2.resize(img, (0,0), fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        rh, rw = self.raw_size
        ows, ohs, ofs_xs, ofs_ys = self.get_overlapping(h, w, rh, rw)
        trans_fn = self.default_normal_trans_fn if trans_fn is None else trans_fn

        # 开始进行高分辨率normal的对齐
        # 1. 对齐row columns
        aligned_rows = []
        mid_normal, mid_ofs_x, mid_ofs_y = None, None, None
        for ofs_y in ofs_ys: # TODO: 目前先align columns，再align rows，后续可以考虑先align rows，再自下而上align columns
            normals = []
            for ofs_x in ofs_xs:
                input_tensor = trans_fn(img[ofs_y:ofs_y+rh, ofs_x:ofs_x+rw])[None].to(self.device) # (1,3,rh,rw)
                normal = normal_fn(input_tensor) # (1,3,rh,rw)
                normal = normal * 2 - 1 # [0,1] -> [-1,1]
                normal = normal / (normal.norm(dim = 1, keepdim = True) + 1e-15)
                normal = normal[0].detach().cpu().numpy() # (3,rh,rw)
                normals.append(normal.transpose(0, 2, 1)) # (3,rw,rh)
                if ofs_y == ofs_ys[len(ofs_ys)//2] and ofs_x == ofs_xs[len(ofs_xs)//2]:
                    mid_normal, mid_ofs_x, mid_ofs_y = normal, ofs_x, ofs_y
                # normals.append(normal_fn(input_tensor)[0].detach().cpu().numpy())
            if mode=='TBLR':
                cur_normal = normals[0]
                for i in range(1,len(normals)):
                    cur_normal = align_normal_y(cur_normal, normals[i], cur_normal.shape[1]-ows[i], cur_normal.shape[1], 0, ows[i])
            elif mode=='MM':
                mid = len(normals) // 2
                cur_normal = normals[mid]
                for i in range(mid+1,len(normals)):
                    cur_normal = align_normal_y(cur_normal, normals[i], cur_normal.shape[1]-ows[i], cur_normal.shape[1], 0, ows[i])
                for i in range(mid-1,-1,-1):
                    cur_normal = align_normal_y(cur_normal, normals[i], 0, ows[i+1], normals[i].shape[1]-ows[i+1], normals[i].shape[1])
            else:
                raise NotImplementedError
            # cur_normal: (3, w, rh)
            aligned_rows.append(cur_normal.transpose(0, 2, 1))
        # 2. 对齐rows
        if mode=='TBLR':
            cur_normal = aligned_rows[0]
            for i in range(1,len(aligned_rows)):
                cur_normal = align_normal_y(cur_normal, aligned_rows[i], cur_normal.shape[1]-ohs[i], cur_normal.shape[1], 0, ohs[i])
        elif mode=='MM':
            mid = len(aligned_rows) // 2
            cur_normal = aligned_rows[mid]
            for i in range(mid+1,len(aligned_rows)):
                cur_normal = align_normal_y(cur_normal, aligned_rows[i], cur_normal.shape[1]-ohs[i], cur_normal.shape[1], 0, ohs[i])
            for i in range(mid-1,-1,-1):
                cur_normal = align_normal_y(cur_normal, aligned_rows[i], 0, ohs[i+1], aligned_rows[i].shape[1]-ohs[i+1], aligned_rows[i].shape[1])
        else:
            raise NotImplementedError
        # cur_normal: (3, h, w), mid_normal: (3, rh, rw)
        # 3. align with mid normal
        R = best_fit_transform(cur_normal[:,mid_ofs_y:mid_ofs_y+rh,mid_ofs_x:mid_ofs_x+rw].reshape(3, -1).T, mid_normal.reshape(3, -1).T)
        cur_normal = (R @ cur_normal.reshape(3, -1)).reshape(cur_normal.shape)
        highres_normal = cur_normal.transpose(1, 2, 0) # (h, w, 3)
        if downscale!=1.0:
            highres_normal = cv2.resize(highres_normal, (origin_w,origin_h), interpolation=cv2.INTER_LINEAR)
            highres_normal = highres_normal / (np.linalg.norm(highres_normal, axis=2, keepdims=True) + 1e-15)
        return highres_normal

    def get_highres_depth(self, img, depth_fn, trans_fn=None, downscale=1.0, mode='MM'):
        # img: (h,w,3), np.array, rgb
        # depth_fn: input: (b,3,rh,rw), output: (b,1,rh,rw), depth_fn output∈[0,1]
        # trans_fn: input: np.array(h,w,3), output: tensor(3,rh,rw), prepare for the prior model
        # mode: 'MM' or 'TBLR', 'MM' means align from middle to both sides, 'TBLR' means align from top to bottom and left to right
        if downscale!=1.0:
            origin_h, origin_w = img.shape[:2]
            img = cv2.resize(img, (0,0), fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        rh, rw = self.raw_size
        ows, ohs, ofs_xs, ofs_ys = self.get_overlapping(h, w, rh, rw)
        trans_fn = self.default_depth_trans_fn if trans_fn is None else trans_fn

        # 开始进行高分辨率depth的对齐
        # 1. 对齐row columns
        aligned_rows = []
        mid_depth, mid_ofs_x, mid_ofs_y = None, None, None
        for ofs_y in ofs_ys:
            depths = []
            for ofs_x in ofs_xs:
                input_tensor = trans_fn(img[ofs_y:ofs_y+rh, ofs_x:ofs_x+rw])[None].to(self.device) # (1,3,rh,rw)
                depth = depth_fn(input_tensor).detach().cpu().clamp(0, 1) # (1,1,rh,rw)
                depths.append(depth.reshape(1, rh, rw).permute(0,2,1)) # (1,rw,rh)
                if ofs_y == ofs_ys[len(ofs_ys)//2] and ofs_x == ofs_xs[len(ofs_xs)//2]:
                    mid_depth, mid_ofs_x, mid_ofs_y = depth.reshape(1, rh, rw), ofs_x, ofs_y
            if mode=='TBLR':
                cur_depth = depths[0]
                for i in range(1,len(depths)):
                    cur_depth = align_y(cur_depth, depths[i], cur_depth.shape[1]-ows[i], cur_depth.shape[1], 0, ows[i])
            elif mode=='MM':
                mid = len(depths) // 2
                cur_depth = depths[mid]
                for i in range(mid+1,len(depths)):
                    cur_depth = align_y(cur_depth, depths[i], cur_depth.shape[1]-ows[i], cur_depth.shape[1], 0, ows[i])
                for i in range(mid-1,-1,-1):
                    cur_depth = align_y(cur_depth, depths[i], 0, ows[i+1], depths[i].shape[1]-ows[i+1], depths[i].shape[1])
            else:
                raise NotImplementedError
            # cur_depth: (1, w, rh)
            aligned_rows.append(cur_depth.permute(0, 2, 1))
        # 2. 对齐rows
        if mode=='TBLR':
            cur_depth = aligned_rows[0]
            for i in range(1,len(aligned_rows)):
                cur_depth = align_y(cur_depth, aligned_rows[i], cur_depth.shape[1]-ohs[i], cur_depth.shape[1], 0, ohs[i])
        elif mode=='MM':
            mid = len(aligned_rows) // 2
            cur_depth = aligned_rows[mid]
            for i in range(mid+1,len(aligned_rows)):
                cur_depth = align_y(cur_depth, aligned_rows[i], cur_depth.shape[1]-ohs[i], cur_depth.shape[1], 0, ohs[i])
            for i in range(mid-1,-1,-1):
                cur_depth = align_y(cur_depth, aligned_rows[i], 0, ohs[i+1], aligned_rows[i].shape[1]-ohs[i+1], aligned_rows[i].shape[1])
        else:
            raise NotImplementedError
        # cur_depth: (1, h, w), mid_depth: (1, rh, rw)
        # 3. align with mid depth
        scale, shift = compute_scale_and_shift(cur_depth[:,mid_ofs_y:mid_ofs_y+rh,mid_ofs_x:mid_ofs_x+rw], mid_depth, torch.ones_like(mid_depth))
        cur_depth = scale * cur_depth + shift
        print(f'estimated depth range: {cur_depth.min():.4f}m - {cur_depth.max():.4f}m, mean: {cur_depth.mean():.4f}m')
        # cur_depth = (cur_depth - cur_depth.min()) / (cur_depth.max() - cur_depth.min() + 1e-15) # TODO: 是否强制align到[0,1]

        highres_depth = cur_depth[0].numpy() # (h, w)
        if downscale!=1.0:
            highres_depth = cv2.resize(highres_depth, (origin_w,origin_h), interpolation=cv2.INTER_LINEAR)
        return highres_depth

