import torch
from models.loss import compute_scale_and_shift
import numpy as np
from PIL import Image
import cv2

def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return [int(h), int(m), int(s)]


def split_input(sample, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    split_keys = ['rays_o', 'rays_d', 'depth_scale', 'normal']
    for i in range(0, total_pixels, n_pixels):
        data = {}
        data['idx'] = sample['idx']
        data['K'] = sample['K']
        data['pose'] = sample['pose']
        for key in split_keys:
            data[key] = sample[key][:, i:i + n_pixels, :] # (B, chunk, -1)
        split.append(data)
    return split

def merge_output(outputs):
    # Merge the split output.
    merge = {}
    for entry in outputs[0]:
        if outputs[0][entry] is None:
            continue
        entry_list = []
        for r in outputs:
            entry_list.append(r[entry])
        merge[entry] = torch.cat([r[entry] for r in outputs],1) # r[entry] shape: (batch_size, chunk, -1)
    return merge


def get_plot_data(outputs, sample, h, w, monocular_depth = True, with_single=False):
    # outputs: {'rgb': (B, h*w, 3), 'depth': (B, h*w, 1), 'normal': (B, h*w, 3)}
    batch_size = outputs['rgb'].shape[0]
    scale_shifts = compute_scale_and_shift(outputs['depth'], sample['depth'])
    plot_outputs = []
    for b in range(batch_size):
        plot_output = {}
        idx = sample['idx'][b].item()  # image index
        rgb = outputs['rgb'][b].cpu().numpy()
        depth = outputs['depth'][b].cpu().numpy()
        normal = outputs['normal'][b].cpu().numpy()
        gt_rgb = sample['rgb'][b].cpu().numpy()
        gt_depth = sample['depth'][b].cpu().numpy()
        gt_normal = sample['normal'][b].cpu().numpy()
        R_c2w = sample['pose'][b].cpu().numpy()[:3, :3]
        # 处理可视化
        # rgb
        rgb_diff = np.abs(rgb - gt_rgb)
        cat_rgb = np.concatenate([rgb.reshape(h, w, 3), gt_rgb.reshape(h, w, 3)], axis=0)
        cat_rgb_diff = np.concatenate([rgb_diff.reshape(h, w, 3), gt_rgb.reshape(h, w, 3)], axis=0)
        plot_rgb = Image.fromarray((cat_rgb * 255).astype(np.uint8))
        plot_rgb_diff = Image.fromarray((cat_rgb_diff * 255).astype(np.uint8))
        # normal
        normal, gt_normal = (normal + 1) / 2, (gt_normal + 1) / 2
        cat_normal = np.concatenate([normal.reshape(h, w, 3), gt_normal.reshape(h, w, 3)], axis=0)
        plot_normal = Image.fromarray((cat_normal * 255).astype(np.uint8))
        # depth
        scale, shift = scale_shifts[b, 0, 0].item(), scale_shifts[b, 1, 0].item()
        if monocular_depth:
            depth = scale * depth + shift
        depth, gt_depth = (depth - depth.min()) / (depth.max() - depth.min()+1e-6), (gt_depth - gt_depth.min()) / (
                    gt_depth.max() - gt_depth.min()+1e-6)
        depth_bgr = cv2.applyColorMap((depth.reshape(h, w) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        gt_depth_bgr = cv2.applyColorMap((gt_depth.reshape(h, w) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cat_depth = np.concatenate([depth_bgr, gt_depth_bgr], axis=0)
        plot_depth = Image.fromarray(cv2.cvtColor(cat_depth, cv2.COLOR_BGR2RGB))
        # merge rgb, rgb_diff, normal, depth
        plot_merge = Image.fromarray(
            np.concatenate([np.array(plot_rgb),np.array(plot_rgb_diff), np.array(plot_normal), np.array(plot_depth)], axis=1))
        # *single rgb、rgb_diff、normal、depth
        if with_single:
            plot_output['single_rgb'] = Image.fromarray((rgb.reshape(h, w, 3) * 255).astype(np.uint8))
            plot_output['single_gt_rgb'] = Image.fromarray((gt_rgb.reshape(h, w, 3) * 255).astype(np.uint8))
            plot_output['single_rgb_diff'] = Image.fromarray((rgb_diff.reshape(h, w, 3) * 255).astype(np.uint8))
            plot_output['single_normal'] = Image.fromarray((normal.reshape(h, w, 3) * 255).astype(np.uint8))
            plot_output['single_depth'] = Image.fromarray(cv2.cvtColor((depth_bgr), cv2.COLOR_BGR2RGB))
            plot_output['single_gt_depth'] = Image.fromarray(cv2.cvtColor((gt_depth_bgr), cv2.COLOR_BGR2RGB))
            plot_output['single_mono_normal'] = Image.fromarray((gt_normal.reshape(h, w, 3) * 255).astype(np.uint8))
        # nbfield related results
        if outputs.get('quat', None) is not None:
            quat = outputs['quat'][b].cpu().numpy() # (h*w, 4), in world space
            half_theta = np.arccos(quat[:, 0]) # ∈ [0, π]
            n = quat[:, 1:] / (np.sin(half_theta)[:, None] + 1e-6) # (h*w, 3)
            n = (R_c2w.T@ n.T).T # n in camera space
            # half_theta = (half_theta - half_theta.min()) / (half_theta.max() - half_theta.min() + 1e-6) # half theta visualized as bias intensity
            plot_half_theta = (half_theta.reshape(h, w, 1)/np.pi*255).astype(np.uint8).repeat(3, axis=2)
            plot_n = ((n.reshape(h, w, 3) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            biased_normal = outputs['biased_normal'][b].cpu().numpy() # normal ---> mono normal
            biased_mono_normal = outputs['biased_mono_normal'][b].cpu().numpy() # mono normal ---> normal
            biased_normal, biased_mono_normal = (biased_normal + 1) / 2, (biased_mono_normal + 1) / 2
            cat_biased = np.concatenate([biased_normal.reshape(h, w, 3), biased_mono_normal.reshape(h, w, 3)], axis=0)
            plot_biased = Image.fromarray((cat_biased * 255).astype(np.uint8))
            plot_output['biased_normal'] = plot_biased
            # *single biased_normal、biased_mono_normal
            if with_single:
                plot_output['single_biased_normal'] = Image.fromarray((biased_normal.reshape(h, w, 3) * 255).astype(np.uint8))
                plot_output['single_biased_mono_normal'] = Image.fromarray((biased_mono_normal.reshape(h, w, 3) * 255).astype(np.uint8))

            # angle between normal and biased_normal
            normal = normal * 2 - 1
            biased_normal = biased_normal * 2 - 1
            gt_normal = gt_normal * 2 - 1
            angle = np.arccos(np.sum(normal * biased_normal, axis=-1)) / np.pi * 180
            angle2 = np.arccos(np.sum(normal * gt_normal, axis=-1)) / np.pi * 180 # angle between normal and gt_normal
            plot_angle = (angle.reshape(h, w, 1) / 180 * 255).astype(np.uint8).repeat(3, axis=2)
            plot_angle2 = (angle2.reshape(h, w, 1) / 180 * 255).astype(np.uint8).repeat(3, axis=2)
            plot_angle_both = np.concatenate([plot_angle, plot_angle2], axis=0)
            plot_quat = Image.fromarray(np.concatenate([plot_n, plot_half_theta, plot_angle], axis=1))
            plot_output['quat'] = plot_quat
            plot_output['angle_both'] = Image.fromarray(plot_angle_both)

            plot_merge_quat = np.concatenate([np.concatenate([plot_n, plot_angle], axis=0), np.array(plot_normal), np.array(plot_biased)],axis=1)
            plot_output['merge_quat'] = Image.fromarray(plot_merge_quat)


            angle_mask1 = ((angle<5).reshape(h,w,1).astype(np.float32)*255).repeat(3,axis=-1).astype(np.uint8) # should smooth
            angle_mask2 = ((angle>30).reshape(h,w,1).astype(np.float32)*255).repeat(3,axis=-1).astype(np.uint8) # should bias
            cat_angle = np.concatenate([angle_mask2, angle_mask1], axis=0)
            plot_output['angle_mask'] = Image.fromarray(cat_angle)

            # *single n、half_theta、angle、biased_normal
            if with_single:
                plot_output['single_n'] = Image.fromarray((plot_n).astype(np.uint8))
                plot_output['single_half_theta'] = Image.fromarray((plot_half_theta).astype(np.uint8))
                plot_output['single_angle'] = Image.fromarray((plot_angle).astype(np.uint8))
                plot_output['single_angle2'] = Image.fromarray((plot_angle2).astype(np.uint8))

            ############################################################# to visualize using plt ##############################
            import matplotlib.pyplot as plt
            # # half_theta_mask = ((half_theta/np.pi*360>30).reshape(h,w,1).astype(np.float32)*255).repeat(3,axis=-1).astype(np.uint8)
            # cv2.imshow('angle_mask1',angle_mask1)
            # cv2.imshow('angle_mask2',angle_mask2)
            # # cv2.imshow('half_theta_mask',half_theta_mask)
            # cv2.waitKey(0)
            # # plt mask
            # # plt.imshow(mask.reshape(h,w))
            # plt.hist(angle.flatten(), bins=range(0, 181, 1), edgecolor='black')
            # for bin in range(0, 181, 1):
            #     plt.axvline(x=bin, color='r')
            # plt.xlabel('angle')
            # plt.ylabel('count')
            # plt.show()

        plot_output['idx'] = idx
        plot_output['rgb'] = plot_rgb
        plot_output['rgb_diff'] = plot_rgb_diff
        plot_output['depth'] = plot_depth
        plot_output['normal'] = plot_normal
        plot_output['merge'] = plot_merge

        plot_outputs.append(plot_output)
    return plot_outputs