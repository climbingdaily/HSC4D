"""
Defines losses used in the optimization
"""
import torch

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.split(os.path.abspath( __file__))[0]))
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as cham

import torchgeometry as tgm
from torch.nn import functional as F
import open3d as o3d
from tools.tool_function import read_json_file

root_path = "/".join(os.path.abspath(__file__).split('/')[:-1])

def load_vertices():
    all_vertices = read_json_file(os.path.join(root_path, 'initialize', 'vertices', 'all_new.json'))
    back = all_vertices['back_new']

    contact_verts = {}
    for part in ['right_toe', 'left_toe', 'left_heel', 'right_heel']:
        contact_verts[part] = np.array(all_vertices[part]['verts'], dtype=np.int32)

    contact_verts['back'] = np.array(all_vertices['back_new']['verts'], dtype=np.int32)
    return contact_verts
    
def find_stable_foot(smpl_verts, translations, pre_smpl = None):
    contact_verts = load_vertices()
    # back['verts']
    new_smpl_verts = smpl_verts.clone().detach() + translations.clone().detach().unsqueeze(1)
    if pre_smpl is not None:
        new_smpl_verts = torch.cat((pre_smpl, pre_smpl), dim = 0)
    left_heels = new_smpl_verts[:, contact_verts['left_heel']]
    left_toes = new_smpl_verts[:, contact_verts['left_toe']]
    right_heels = new_smpl_verts[:, contact_verts['right_heel']]
    right_toes = new_smpl_verts[:, contact_verts['right_toe']]

    lh_move = (left_heels[1:] - left_heels[:-1]).mean(dim=1).norm(dim=1)
    lt_move = (left_toes[1:] - left_toes[:-1]).mean(dim=1).norm(dim=1)
    rh_move = (right_heels[1:] - right_heels[:-1]).mean(dim=1).norm(dim=1)
    rt_move = (right_toes[1:] - right_toes[:-1]).mean(dim=1).norm(dim=1)

    left_foot = torch.cat((left_heels, left_toes), dim=1) 
    right_foot = torch.cat((right_heels, right_toes), dim=1)
    
    lfoot_move = (left_foot[1:] - left_foot[:-1]).mean(dim=1).norm(dim=1)
    rfoot_move = (right_foot[1:] - right_foot[:-1]).mean(dim=1).norm(dim=1)

    lhp = left_heels.mean(dim=1)[:,2]
    ltp = left_toes.mean(dim=1)[:,2]
    rhp = right_heels.mean(dim=1)[:,2]
    rtp = right_toes.mean(dim=1)[:,2]
    lhp -= lhp[0].item()
    ltp -= ltp[0].item()
    rhp -= rhp[0].item()
    rtp -= rtp[0].item()

    # fig = plt.figure(1)
    # plt.scatter(np.arange(left_heels.shape[0]), lhp.detach().cpu().numpy(), label='left heel')
    # plt.scatter(np.arange(left_heels.shape[0]), ltp.detach().cpu().numpy(), label='left toe')
    # plt.scatter(np.arange(left_heels.shape[0]), rhp.detach().cpu().numpy(), label='right heel')
    # plt.scatter(np.arange(left_heels.shape[0]), rtp.detach().cpu().numpy(), label='right toe')
    # plt.plot(np.arange(lh_move.shape[0]), (rfoot_move).detach().cpu().numpy()/2, label='right foot')
    # plt.plot(np.arange(lh_move.shape[0]), (lfoot_move).detach().cpu().numpy()/2, label='left foot')
    # plt.xlabel('Frames')
    # plt.ylabel('Distance between two frames (m)')
    # plt.legend()
    # plt.show()
    
    states = []
    for i in range(lt_move.shape[0]):
        # If both feet's moving distance < 2cm, set as stable
        if rfoot_move[i] <= 0.01 and lfoot_move[i] <= 0.01:
            states.append(0)
        elif lfoot_move[i] <= 0.01:
            states.append(-1) # left foot stable
        elif rfoot_move[i] <= 0.01:
            states.append(1)   # right foot stable
        else:
            states.append(-2) # bad case (foot sliding)
    states = np.asarray(states, dtype=np.int32)

    while True:
        count = 0
        for i in range(2, states.shape[0]-2):
            if states[i-2] == states[i-1] and states[i+1] == states[i+2] and states[i] != states[i+1] and states[i] != states[i-1]:
                states[i] = states[i-1]
                count += 1
        if count == 0:
            break

    while True:
        count = 0
        for i in range(2, states.shape[0]-2):
            if states[i] != states[i+1] and states[i] != states[i-1]:
                states[i] = states[i-1]
                count += 1
        if count == 0:
            break

    # for idx, s in enumerate(states):
    # plt.plot(states)
    # plt.show()
    if pre_smpl is None:
        states = np.concatenate((np.asarray(states[:1]), states))
        lfoot_move = torch.cat((lfoot_move[:1].clone(), lfoot_move))
        rfoot_move = torch.cat((rfoot_move[:1].clone(), rfoot_move))
        lh_move = torch.cat((lh_move[:1].clone(), lh_move))
        lt_move = torch.cat((lt_move[:1].clone(), lt_move))
        rh_move = torch.cat((rh_move[:1].clone(), rh_move))
        rt_move = torch.cat((rt_move[:1].clone(), rt_move))
    lfoot_move = lfoot_move.detach().cpu().numpy()
    rfoot_move = rfoot_move.detach().cpu().numpy()
    lh_move = lh_move.detach().cpu().numpy()
    lt_move = lt_move.detach().cpu().numpy()
    rh_move = rh_move.detach().cpu().numpy()
    rt_move = rt_move.detach().cpu().numpy()
    # return states, lfoot_move, rfoot_move, lh_move, lt_move, rh_move, rt_move
    return states, lfoot_move, rfoot_move

def trans_imu_smooth(trans_params, imu_trans, mode='XY'):
    """
    :param trans_params:
    :param imu_trans:
    :return:
    """
    if mode == 'XY':
        select = [0, 1]
    elif mode == 'XYZ':
        select = [0, 1, 2]
    else:
        select = [0, 1, 2]
    trans_diffs = torch.norm(trans_params[:-1,select] - trans_params[1:, select], dim =1)
    imu_diffs = torch.norm(imu_trans[:-1,select] - imu_trans[1:, select], dim =1)
    diffs = trans_diffs - imu_diffs

    diffs_new2 = torch.nn.functional.relu(diffs)
    return torch.mean(diffs_new2)

def joint_orient_error(pred_mat, gt_mat):
    """
    Find the orientation error between the predicted and GT matrices
    Args:
        pred_mat: Batch x 3 x 3
        gt_mat: Batch x 3 x 3
    Returns:

    """
    r1 = pred_mat
    r2t = torch.transpose(gt_mat, 2, 1)
    r = torch.bmm(r1, r2t)

    pad_tensor = F.pad(r, [0, 1])
    residual = tgm.rotation_matrix_to_angle_axis(pad_tensor)
    norm_res = torch.norm(residual, p=2, dim=1)

    return torch.mean(norm_res)

def sliding_constraint(smpl_verts, foot_states, jump_list, lfoot_move = None, rfoot_move = None):
    contact_verts = load_vertices()
    # vertex_diffs = smpl_verts[:-1] - smpl_verts[1:]
    # vertex_diffs = vertex_diffs.reshape(-1, 3)
    # valid_vertex_diffs = vertex_diffs[frame_verts, :]
    # normed_vertex_diffs = torch.norm(valid_vertex_diffs,  p = 2, dim = 1)

    # _min_move = np.array([lfoot_move, rfoot_move]).min(axis=0)
    min_move = []
    right_foot = np.concatenate((contact_verts['right_heel'], contact_verts['right_toe']))
    left_foot = np.concatenate((contact_verts['left_heel'], contact_verts['left_toe']))
    feet = np.concatenate((left_foot, right_foot))

    valid_vertex_diffs = torch.empty(0,3).to(smpl_verts.device)

    if lfoot_move is not None and rfoot_move is not None:
        for i in range(len(foot_states)):
            if i==0:
                continue
            if lfoot_move[i] > rfoot_move[i]:
                vertex_diffs = smpl_verts[i, right_foot] - smpl_verts[i-1, right_foot]
                min_move += [rfoot_move[i]] * len(right_foot)
            else:
                vertex_diffs = smpl_verts[i, left_foot] - smpl_verts[i-1, left_foot] 
                min_move += [lfoot_move[i]] * len(left_foot)

            vertex_diffs = vertex_diffs.reshape(-1, 3)
            valid_vertex_diffs = torch.cat((valid_vertex_diffs, vertex_diffs))
    else:
        # cur_idx = -1
        slides = 0
        pre_state = foot_states[0]
        pre_stable_foot = feet

        for i, cur_state in enumerate(foot_states):
            if i == 0:
                continue
            if cur_state == pre_state and cur_state != -2:
                if cur_state == 1:
                    vertex_diffs = smpl_verts[i, right_foot] - smpl_verts[i-1, right_foot] 
                    pre_stable_foot = right_foot
                elif cur_state == -1:
                    vertex_diffs = smpl_verts[i, left_foot] - smpl_verts[i-1, left_foot] 
                    pre_stable_foot = left_foot
                elif cur_state == 0:
                    vertex_diffs = smpl_verts[i, feet] - smpl_verts[i-1, feet] 
                slides += 1
                vertex_diffs = vertex_diffs.reshape(-1, 3)
                # normed_vertex_diffs = torch.norm(vertex_diffs,  p = 2, dim = 1)
                valid_vertex_diffs = torch.cat((valid_vertex_diffs, vertex_diffs))

            pre_state = cur_state

    if valid_vertex_diffs.shape[0] > 0:
        min_move = torch.from_numpy(np.array(min_move)).type(torch.FloatTensor).cuda()
        loss = F.relu(torch.norm(valid_vertex_diffs,  p = 2, dim = 1) - min_move)
        return loss.mean(), len(foot_states) - 1
    else:
        return None, slides

def contact_constraint(scene_segmentations, smpl_verts, foot_states, jump_list):
    contact_verts = load_vertices()
    
    batch_verts = np.empty(0)
    batch_equations = np.empty((0, 4))
    # foot_states = np.concatenate((np.asarray(foot_states[:1]), foot_states)) # foot_statas.shape is smaller 1 than batch size
    
    batch_planes = o3d.geometry.PointCloud()

    num_contact = 0
    foot_states = np.array(foot_states)
    for i in range(foot_states.shape[0]):
        if i in jump_list:
            continue
        # equations, planes, planes_idx, rest_idx = scene_segmentations[i]
        # if len(planes) < 1 or equations[0][2] < 0.95:
        #     continue
        planes = scene_segmentations[i]

        if foot_states[i] == 1:
            foot = 'right' 
        elif foot_states[i] == -1:
            foot = 'left'
        elif foot_states[i] == 0:
            
            left_heels = smpl_verts[i, contact_verts['left_heel']]
            left_toes = smpl_verts[i, contact_verts['left_toe']]
            right_heels = smpl_verts[i, contact_verts['right_heel']]
            right_toes = smpl_verts[i, contact_verts['right_toe']]

            left_foot = torch.cat((left_heels, left_toes), dim=0) 
            right_foot = torch.cat((right_heels, right_toes), dim=0)
            z_lf = left_foot.mean(dim=0)[2]
            z_rf = right_foot.mean(dim=0)[2]
            foot = 'right' if z_rf < z_lf else 'left'
        else:
            'not walkking'
            continue
        z_postion_h = smpl_verts[i, contact_verts[foot + '_heel']].mean(dim=0)[2]
        z_postion_t = smpl_verts[i, contact_verts[foot + '_toe']].mean(dim=0)[2]
        part = '_heel' if z_postion_h < z_postion_t else '_toe'
        verts_num = len(contact_verts[foot + part])

        lowest_position = smpl_verts[i, contact_verts[foot + part]].mean(dim=0)[2]

        batch_planes += planes
        batch_verts = np.concatenate((batch_verts, contact_verts[foot + part] + 6890 * i))
        num_contact += 1
        # ================== choose planes ===============
        # dist_foot_to_plane = 100
        # plane_id = -1
        # for i, p in enumerate(planes):
        #     if equations[i][2] >= 0.95:
        #         z_dist = abs(p.get_center()[2] - lowest_position)  # plane to foot distance
        #         if z_dist < 0.6 and z_dist < dist_foot_to_plane:
        #             dist_foot_to_plane = z_dist
        #             plane_id = i
        # if plane_id > -1:
        #     batch_verts = np.concatenate((batch_verts, contact_verts[foot + part] + 6890 * i))
        #     batch_equations = np.concatenate((batch_equations, equations[plane_id].reshape(1,-1).repeat(verts_num, axis=0)))
        #     batch_planes += planes[plane_id]

    # if(batch_equations.shape[0] > 0):
    #     device = smpl_verts.device
    #     smpl_verts = smpl_verts.reshape(-1, 3)
    #     verts = smpl_verts[batch_verts, :].reshape(-1,3)

    #     # ============ point to plane distance ============
    #     # verts_1 = torch.cat((verts, torch.ones(verts.shape[0], 1).to(device)), dim=1)
    #     # batch_equations = torch.from_numpy(batch_equations).type(torch.FloatTensor).to(device)
    #     # loss = abs(verts_1.mul(batch_equations).sum(dim=1)).mean()
    #     return loss
    # else:
    #     return None
    
    # ============ CD distance ==============
    if batch_planes.has_points():
        device = smpl_verts.device
        distChamfer = cham.chamfer_3DDist()
        batch_planes = batch_planes.voxel_down_sample(voxel_size=0.01)
        batch_planes_points = torch.from_numpy(np.asarray(batch_planes.points)[None, :]).type(torch.FloatTensor).to(device)
        smpl_verts = smpl_verts.reshape(-1, 3)
        verts = smpl_verts[batch_verts, :].reshape(-1,3)
        loss, _, _, _ = distChamfer(verts.unsqueeze(0).contiguous(), batch_planes_points)
        torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
        loss = torch.sqrt(loss).mean()
        return loss, num_contact
    else:
        return None, 0