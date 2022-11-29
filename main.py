################################################################################
# File: /main.py                                                               #
# Created Date: Tuesday July 26th 2022                                         #
# Author: climbingdaily                                                        #
# -----                                                                        #
# Modified By: the developer climbingdaily at yudidai@stu.xmu.edu.cn           #
# https://github.com/climbingdaily                                             #
# -----                                                                        #
# Copyright (c) 2022 yudidai                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
################################################################################

import os
import time
import pickle as pkl
import torch
import numpy as np

from tensorboardX import SummaryWriter
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from tools.tool_function import TRANSFORM_MOCAP, load_scene, multiprocess_save_smpl_model, read_json_file, mocap_to_smpl_axis, save_json_file, save_traj_in_dir, rigid_transform_3D
from losses import contact_constraint, trans_imu_smooth, joint_orient_error, sliding_constraint, find_stable_foot
# from tools.segmentation import Segmentation

class Optimizer():
    def __init__(self, args):
        self.optimize = True
        self.is_cuda = True
        self.args = args
        self.rot_offset = torch.eye(3)
        self.get_args()

    def set_cuda(self, is_cuda):
        self.is_cuda = is_cuda

    def set_lidar_offset(self):
        meta_init = read_json_file(self.init_json)
        lidar_offset = np.asarray(meta_init['lidar_offset'])
        print('Lidar offset (in lidar coordinate): ', np.round(lidar_offset, 3))
        self.lidar_offset = torch.from_numpy(lidar_offset)
        self.lidar_offset.requires_grad = False
        if self.is_cuda:
            self.lidar_offset = self.lidar_offset.type(torch.FloatTensor).cuda()

    def set_lidar_traj(self, lidar_traj_path):
        self.synced_lidar = np.loadtxt(lidar_traj_path)
        if self.synced_lidar.shape[0] != self.synced_mocap_rots_axis.shape[0]:
            return False
        else:
            return True

    def get_args(self):
        dataset_root = self.args.dataset_root
        data_name = self.args.data_name
        print('Loading arguments...')

        self.data_dir = os.path.join(dataset_root, data_name + '_data')
        os.makedirs(self.data_dir, exist_ok=True)

        self.init_json = os.path.join(self.data_dir, data_name + '_init.json')
        
        # =====================        
        # Load synced lidar and mocap data
        # =====================
        save_synced_data = os.path.join(self.data_dir, 'synced_data_for_optimization.pkl')
        with open(save_synced_data, 'rb') as f:
            synced_data = pkl.load(f)
            print('Load data in ', save_synced_data)
        self.synced_mocap_trans = np.matmul(synced_data['synced_pos'], TRANSFORM_MOCAP.T) # x, y, z
        self.synced_lidar = synced_data['synced_lidar'] # idx, X, Y, Z, qx, qy, qz, qw, timestamp
        if 'synced_rot_axis' in synced_data.keys():
            self.synced_mocap_rots_axis = synced_data['synced_rot_axis']
        else:
            self.synced_mocap_rots_axis = mocap_to_smpl_axis(read_json_file(
                './initialize/mocap_init.json'), synced_data['synced_rot']) # eular angles, 'YXZ'-> axis
            synced_data['synced_rot_axis'] = self.synced_mocap_rots_axis
            with open(save_synced_data, 'wb') as f:
                pkl.dump(synced_data, f)
        
        self.opt_start = int(max(self.args.opt_start - self.synced_lidar[0, 0], 0))
        self.opt_end = self.args.opt_end - self.args.opt_start + self.opt_start
        if self.opt_end < 0 or self.opt_end >= self.synced_lidar.shape[0]:
            self.opt_end = self.synced_lidar.shape[0] - 1

        # =====================
        # initialize smpl model
        # =====================
        # self.betas = data_from_json(dataset_root, 'betas')
        root_path = "/".join(os.path.abspath(__file__).split('/')[:-1])
        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            # gender=data_from_json(dataset_root, 'gender')
            model_root=os.path.join(root_path, 'smpl'))
        self.faces = self.smpl_layer.th_faces
        
        # =====================
        # mocap coordinate to lidar coordinate (Z-axis forward ===> Y-axis forward)
        # =====================
        self.transform_mocap = torch.from_numpy(TRANSFORM_MOCAP)
        self.transform_mocap.requires_grad = False

        self.set_lidar_offset()        

        if self.is_cuda:
            self.smpl_layer.cuda()
            self.transform_mocap = self.transform_mocap.type(torch.FloatTensor).cuda()

        self.set_foot_states()

        self.save_str = f'{self.opt_start + self.synced_lidar[0,0]:.0f}_{self.opt_end + self.synced_lidar[0,0]:.0f}_V{self.args.wt_ft_vel:.0f}_F{self.args.wt_ft_cont:.0f}_M{self.args.wt_trans_imu_smth:.0f}_R{self.args.radius:.1f}_W{self.args.window_frames}'
        self.writer = SummaryWriter(os.path.join(
            dataset_root, 'runs', time.strftime("%m_%d_%H_%M_", time.localtime()) + self.save_str))

    def set_foot_states(self):
        start = 0
        end = self.opt_end
        # smpl_models = torch.empty((0, 6890, 3)).cuda()

        mocap_trans = torch.empty(0).cuda()
        self.foot_states = np.empty(0)
        self.lfoot_move = np.empty(0)
        self.rfoot_move = np.empty(0)
        # mocap_rots = torch.tensor(self.synced_mocap_rots_axis).type(torch.FloatTensor).cuda()
        # mocap_trans = torch.tensor(self.synced_mocap_trans).type(torch.FloatTensor).cuda()
        for i in range(start, end, 500):
            s_idx = i
            e_idx = 500 + i
            if e_idx > end:
                e_idx = end
            if s_idx > self.synced_lidar.shape[0]:
                break
            _, _, mocap_trans_params, mocap_ori_params, mocap_after_ori_params = self.init_params(s_idx, e_idx)
            mocap_rots = torch.cat([mocap_ori_params, mocap_after_ori_params], dim=1)

            smpl_verts, _ = self.mocap_to_smpl(mocap_rots, mocap_trans_params)
            # smpl_models = torch.cat((smpl_models, smpl_verts))
            # mocap_trans = torch.cat((mocap_trans, mocap_trans_params))

            foot_states, lfoot_move, rfoot_move = find_stable_foot(smpl_verts, mocap_trans_params - mocap_trans_params)

            self.foot_states = np.concatenate((self.foot_states, foot_states))
            self.lfoot_move = np.concatenate((self.lfoot_move, lfoot_move))
            self.rfoot_move = np.concatenate((self.rfoot_move, rfoot_move))


    def init_params(self, start, end, delta_trans=0):
        if start > self.synced_lidar.shape[0]:
            print('======================================================')
            print('Start idx is larger than data lenght\n')
            print('======================================================')
            return None
        if end > self.synced_lidar.shape[0]:
            end = self.synced_lidar.shape[0]
        lidar_ori_params = torch.from_numpy(self.synced_lidar[start: end, 4:8])
        lidar_trans_params = torch.from_numpy(self.synced_lidar[start: end, 1:4])
        mocap_pose_params = torch.from_numpy(self.synced_mocap_rots_axis[start: end])
        mocap_trans_params = torch.from_numpy(self.synced_mocap_trans[start: end])
        mocap_ori_params = mocap_pose_params[:,:3]
        mocap_after_ori_params = mocap_pose_params[:,3:]
        rot_offset, _ = rigid_transform_3D(
            self.synced_mocap_trans[start: end].T, self.synced_lidar[start: end, 1:4].T)
        if rot_offset is not None:
            self.rot_offset = torch.from_numpy(rot_offset)
        lidar_ori_params.requires_grad = False
        lidar_trans_params.requires_grad = False
        # mocap_pose_params.requires_grad = False
        mocap_trans_params.requires_grad = False
        mocap_ori_params.requires_grad = False
        mocap_after_ori_params.requires_grad = False
        self.rot_offset.requires_grad = False
        
        if self.is_cuda:
            lidar_ori_params = lidar_ori_params.type(torch.FloatTensor).cuda()
            lidar_trans_params = lidar_trans_params.type(torch.FloatTensor).cuda()
            mocap_trans_params = mocap_trans_params.type(torch.FloatTensor).cuda()
            mocap_ori_params = mocap_ori_params.type(torch.FloatTensor).cuda()
            mocap_after_ori_params = mocap_after_ori_params.type(torch.FloatTensor).cuda()
            self.rot_offset = self.rot_offset.type(torch.FloatTensor).cuda()
            # betas_torch = torch.tensor(self.betas, requires_grad=False).unsqueeze(0).type(torch.FloatTensor).cuda()
        # mocap_trans_params = torch.matmul(mocap_trans_params, self.transform_mocap)
        return lidar_ori_params, lidar_trans_params, mocap_trans_params, mocap_ori_params, mocap_after_ori_params

    def divide_traj_in_number(self, skip=100):
        opt_end = self.opt_end
        init_data = read_json_file(self.init_json)
        a = init_data['lidar_sync']
        b = init_data['mocap_sync']
        f_jump = init_data['lidar_sync'][0]
        jump_list = []
        for i in range(len(b)):
            jump_time = b[i] - b[0] + f_jump
            tt = abs(self.synced_lidar[:,-1] - jump_time)
            jump = np.where( tt == tt.min())[0][0]
            jump_list += np.arange(
                jump - init_data['lidar_framerate'], jump + init_data['lidar_framerate']).astype(np.int64).tolist()
            
        sub_segment_idx = np.arange(self.opt_start, self.opt_end, self.args.window_frames).tolist() 
        if self.opt_end - sub_segment_idx[-1] < self.args.window_frames//2:
            sub_segment_idx[-1] = self.opt_end
        else:
            sub_segment_idx.append(self.opt_end)

        return sub_segment_idx, jump_list

    def mocap_to_smpl(self, pose_params, ldiar_trans_params = None):
        """
        :param pose_params:
        :param trans_params:
        :param ori_params:
        :return:
        """
        # betas_torch_batch = torch.tensor(self.betas, requires_grad=False).unsqueeze(0).repeat(pose_params.shape[0], 1)

        smpl_verts, _, orientations = self.smpl_layer(
            th_pose_axisang=pose_params.cuda()
            # th_betas=betas_torch_batch.cuda()
        )
        # lidar_ori = quaternion_to_matrix(lidar_ori_params)
        # rlidar_offset = torch.matmul(lidar_ori, self.lidar_offset)
        # mocap_trans = (ldiar_trans_params + self.lidar_offset).unsqueeze(1)

        rep_init_T = self.transform_mocap.T.unsqueeze(0).repeat(pose_params.shape[0],1,1)
        if ldiar_trans_params is not None:
            smpl_verts = (smpl_verts @ rep_init_T) @ self.rot_offset.T + ldiar_trans_params.unsqueeze(1)
        else:
            smpl_verts = (smpl_verts @ rep_init_T) @ self.rot_offset.T
        return smpl_verts, orientations

    def save_smpl_models(self, step):
        start = self.opt_start
        end = self.opt_end
        smpl_models = np.empty((0, 6890, 3))
        for i in range(start, end, 1000):
            s_idx = i
            e_idx = 1000 + i
            if e_idx > end:
                e_idx = end
            if s_idx > self.synced_lidar.shape[0]:
                break
            _, lidar_trans_params, _, mocap_ori_params, mocap_after_ori_params = self.init_params(s_idx, e_idx)
            mocap_rots = torch.cat([mocap_ori_params, mocap_after_ori_params], dim=1)

            self.rot_offset[0,0] = 1.
            self.rot_offset[0,1] = 0.
            self.rot_offset[1,0] = 0.
            self.rot_offset[1,1] = 1.
            smpl_verts, _ = self.mocap_to_smpl(mocap_rots, lidar_trans_params)
            

            smpl_models = np.concatenate((smpl_models, smpl_verts.detach().cpu().numpy()))
        start_num = self.synced_lidar[0,0]
        print(f'Saving frams {start + start_num:.0f} to {end + start_num:.0f}')
        multiprocess_save_smpl_model(
            self.synced_lidar[0, 0], smpl_models, self.args.dataset_root, self.args.data_name, step, '_no_opt')
    
    def crop_scene(self, positions):
        # segmentaion = Segmentation()
        grids_segmentaion = []

        for i, p in enumerate(positions):
            p = p.detach().cpu().numpy()
            p -= np.array([0 , 0, 0.8])
            print(f'\rCrop and segment the planes {i+1}/{positions.shape[0]}', end='', flush=True)
            [_, idx, _] = self.scene_kdtree.search_radius_vector_3d(p, radius = self.args.radius)
            grid = self.scene_pcd.select_by_index(list(idx))
            grid = grid.voxel_down_sample(voxel_size=0.02)
            grid.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
            
            # segmentaion.set_pcd(grid)
            # equations, planes, planes_idx, rest_idx = segmentaion.run(
            #     max_plane=8, distance_thresh=self.args.plane_thresh)
            # grids_segmentaion.append([equations, planes, planes_idx, rest_idx])
            grids_segmentaion.append(grid)
        print('')
        return grids_segmentaion

    def cal_mocap_loss(self, new_traj_file = None):
        # 根据距离优化一段轨迹
        self.sub_segment_idx, self.jump_list = self.divide_traj_in_number()

        if new_traj_file:
            new_traj = np.loadtxt(new_traj_file)[:,1:4]
        else:
            new_traj = self.synced_mocap_trans
            # new_traj = np.matmul(new_traj, TRANSFORM_MOCAP)
        new_traj_params = torch.from_numpy(new_traj)
        new_traj_params.requires_grad = False
        if self.is_cuda:
            new_traj_params = new_traj_params.type(torch.FloatTensor).cuda()

        start = self.opt_start
        end = self.opt_end

        smpl_models = torch.empty((0, 6890, 3)).cuda()
        lidar_trans = torch.empty(0).cuda()
        mocap_trans = torch.empty(0).cuda()
        for i in range(start, end, 1000):
            s_idx = i
            e_idx = 1000 + i
            if e_idx > end:
                e_idx = end
            if s_idx > self.synced_lidar.shape[0]:
                break

            _, _, mocap_trans_params, mocap_ori_params, mocap_after_ori_params = self.init_params(s_idx, e_idx)

            mocap_rots = torch.cat([mocap_ori_params, mocap_after_ori_params], dim=1)
            smpl_verts, _ = self.mocap_to_smpl(mocap_rots, new_traj_params[s_idx:e_idx])
            smpl_models = torch.cat((smpl_models, smpl_verts))
            lidar_trans = torch.cat((lidar_trans, new_traj_params[s_idx:e_idx]))
            mocap_trans = torch.cat((mocap_trans, mocap_trans_params))
        
        jump_list = []
        for jl in self.jump_list:
            if jl >= start and jl < end:
                jump_list.append(jl - start)
        
        # foot_states, lfoot_move, rfoot_move = find_stable_foot(smpl_models, mocap_trans - lidar_trans)
        foot_states = self.foot_states[start:end]
        lfoot_move = self.lfoot_move[start:end]
        rfoot_move = self.rfoot_move[start:end]
        sliding_loss, sliding_num = sliding_constraint(smpl_models, foot_states, jump_list, lfoot_move, rfoot_move)
        print('===============================================')
        print(f'Mocap sliding loss: {100  * sliding_loss.item():.3f}, num frames: {sliding_num}')
        print('===============================================')
        
        self.scene_pcd, self.scene_kdtree = load_scene(os.path.dirname(self.args.dataset_root), self.args.data_name)
        scene_grids_segmentations = self.crop_scene(lidar_trans)
        contac_loss, contac_num = contact_constraint(scene_grids_segmentations, smpl_models, foot_states, jump_list)
        print('===============================================')
        print(f'Mocap contac_loss: {100 * contac_loss.item():.3f}, num frames: {contac_num}')
        print('===============================================')

    def run(self):

        self.sub_segment_idx, self.jump_list = self.divide_traj_in_number()

        self.scene_pcd, self.scene_kdtree = load_scene(os.path.dirname(self.args.dataset_root), self.args.data_name)

        num_segs = len(self.sub_segment_idx) - 1
        lcont = np.zeros(num_segs)
        flcont = np.zeros(num_segs)
        lsld = np.zeros(num_segs)
        flsld = np.zeros(num_segs)
        lsmt = np.zeros(num_segs)
        flsmt = np.zeros(num_segs)
        trans_num = 0
        sliding_num = 1
        contact_num = 1
        loss_dict = {}

        delta_trans = 0
        delta_rot = 0
        pre_lidar_trans = 0
        pre_state = 0
        pre_smpl_verts = 0

        for i in range(num_segs):
            start = self.sub_segment_idx[i]
            end = self.sub_segment_idx[i+1]
            loss_dict[i] = {}
            loss_dict[i]['start'] = int(start + self.synced_lidar[0,0])
            loss_dict[i]['end'] = int(end - 1 + + self.synced_lidar[0,0])
            
            # define optimization params
            lidar_ori_params, lidar_trans_params, mocap_trans_params, mocap_ori_params, mocap_after_ori_params = self.init_params(start, end, delta_trans)

            foot_states = self.foot_states[start: end]
            lfoot_move = self.lfoot_move[start: end]
            rfoot_move = self.rfoot_move[start: end]
            
            jump_list = []
            for jl in self.jump_list:
                if jl >= start and jl < end:
                    jump_list.append(jl - start)
            
            scene_grids_segmentations = self.crop_scene(lidar_trans_params)

            if self.args.opt_vars == "trans":
                lidar_trans_params.requires_grad = True
                optimizer = torch.optim.Adam([lidar_trans_params], self.args.learn_rate, betas=(0.9, 0.999))

            elif self.args.opt_vars == "trans_glob":
                lidar_trans_params.requires_grad = True
                mocap_ori_params.requires_grad = True
                optimizer = torch.optim.Adam([lidar_trans_params, mocap_ori_params], self.args.learn_rate, betas=(0.9, 0.999))

            elif self.args.opt_vars == "trans_allpose":
                lidar_trans_params.requires_grad = True
                mocap_ori_params.requires_grad = True
                mocap_after_ori_params.requires_grad = True
                optimizer = torch.optim.Adam([lidar_trans_params, mocap_ori_params, mocap_after_ori_params], self.args.learn_rate, betas=(0.9, 0.999))

            else:
                print('Optimization type error!\n')
                exit()
            
            time_start = time.time()
            mocap_rots = torch.cat([mocap_ori_params, mocap_after_ori_params], dim=1)
            if torch.any(torch.isnan(mocap_rots)):
                print('mocap_rots nan')
            pre_loss = -1
            first_contact = True
            first_slide = True
            first_trans = True
            first_slide = True
            init_first_trans = True
            init_first_slide = True
            for iters in range(self.args.iterations):
                print_str = f'{i+1}/{num_segs}|{start + self.synced_lidar[0,0]:.0f} {end-1 + self.synced_lidar[0,0]:.0f}|{iters} '

                time1 = time.time()
                optimizer.zero_grad()
                sum_loss = 0.0

                smpl_verts, orientations = self.mocap_to_smpl(mocap_rots, lidar_trans_params)

                if torch.any(torch.isnan(smpl_verts)):
                    print('smpl_verts nan')
                if torch.any(torch.isnan(mocap_rots)):
                    print('mocap_rots nan')

                if self.args.wt_ft_cont >=0:
                    contact_loss, cnum = contact_constraint(scene_grids_segmentations, smpl_verts, foot_states, jump_list)
                    if contact_loss:
                        if first_contact:
                            contact_num += cnum
                            flcont[i] = contact_loss.item() * cnum * 100
                            first_contact = False
                        lcont[i] = contact_loss.item() * cnum * 100
                        sum_loss += self.args.wt_ft_cont * contact_loss
                        print_str += f'Foot {contact_loss.item() * 100:.3f} '
                        self.writer.add_scalars(f'window/{i}_{start}_{end}', {'contact':contact_loss.item() * 100}, iters)

                if self.args.wt_ft_vel >=0:
                    vel_loss, snum = sliding_constraint(smpl_verts, foot_states, jump_list, lfoot_move, rfoot_move)
                    if vel_loss:
                        if first_slide:
                            sliding_num += snum
                            flsld[i] = vel_loss.item() * snum * 100
                            first_slide = False
                        lsld[i] = vel_loss.item() * snum * 100
                        sum_loss += self.args.wt_ft_vel* vel_loss
                        print_str += f'Sld {vel_loss.item() * 100:.3f} '
                        self.writer.add_scalars(f'window/{i}_{start}_{end}', {'sliding':vel_loss.item() * 100}, iters)

                if self.args.wt_trans_imu_smth >=0:
                    trans_imu_smth_loss = trans_imu_smooth(
                        lidar_trans_params.squeeze(1), mocap_trans_params, args.imu_smt_mode)
                    tnum = lidar_trans_params.shape[0] - 1
                    if first_trans:
                        trans_num += tnum
                        flsmt[i] = trans_imu_smth_loss.item() * tnum * 100
                        first_trans = False
                    lsmt[i] = trans_imu_smth_loss.item() * tnum * 100
                    sum_loss += self.args.wt_trans_imu_smth * trans_imu_smth_loss
                    print_str += f'Trans {trans_imu_smth_loss.item() * 100:.4f} '
                    self.writer.add_scalars(f'window/{i}_{start}_{end}', {'trans':trans_imu_smth_loss.item() * 100}, iters)

                if self.args.wt_rot_smth >0:
                    glob_smooth_loss = joint_orient_error(
                        orientations[:-1, 0, :, :], orientations[1:, 0, :, :])   
                    sum_loss += self.args.wt_rot_smth * glob_smooth_loss
                    print_str += f'Rot {glob_smooth_loss.item() * 100:.3f} '
                    self.writer.add_scalars(f'window/{i}_{start}_{end}', {'rot':glob_smooth_loss.item() * 100}, iters)
                
                # penetration loss

                if self.args.wt_pose_prior >0:
                    pose_params_cons = torch.tensor(self.synced_mocap_rots_axis[start:end], requires_grad = False).type(torch.FloatTensor).cuda()

                    mocap_after_ori_params_cons = pose_params_cons[:, 3:]

                    diff1 = mocap_after_ori_params - mocap_after_ori_params_cons

                    pose_loss = torch.mean(torch.norm(diff1, p = 2, dim = 1),  dim = 0)
                    print_str += f'Pose {pose_loss.item()*1000:.3f} '
                    sum_loss += self.args.wt_pose_prior * pose_loss
                    self.writer.add_scalars(f'window/{i}_{start}_{end}', {'pose:':pose_loss.item() * 1000}, iters)

                if i != 0:
                    pre_idx = self.sub_segment_idx[i] - 1
                    cur_idx = self.sub_segment_idx[i]
                    if self.args.init_trans_weight >=0:
                        init_trans_diff = pre_lidar_trans - lidar_trans_params[0]
                        init_trans_norm = torch.norm(init_trans_diff, p = 2)

                        imu_imu_diff = torch.tensor(self.synced_mocap_trans[cur_idx] - self.synced_mocap_trans[cur_idx - 1]).type(torch.FloatTensor).cuda()
                        # imu_imu_diff = torch.matmul(imu_imu_diff, self.transform_mocap)
                        init_imu_norm = torch.norm(imu_imu_diff, p = 2)
                        init_trans_loss = torch.nn.functional.relu(init_trans_norm - init_imu_norm)

                        if init_trans_loss:
                            if init_first_trans:
                                flsmt[i] += trans_imu_smth_loss.item() * 100
                                trans_num += 1
                                init_first_trans = False
                            # lsmt[i] += init_trans_loss.item() * 100
                        sum_loss += self.args.init_trans_weight * init_trans_loss

                        print_str += f'Itrans {init_trans_loss.item()*100:.4f} '

                    if self.args.init_vel_weight >=0:
                        init_verts = torch.cat((pre_smpl_verts, smpl_verts[0:1]))
                        # lm, rm 也要cat
                        init_vel_loss, num = sliding_constraint(
                            init_verts, self.foot_states[cur_idx - 1:cur_idx + 1], jump_list, self.lfoot_move[cur_idx - 1: cur_idx + 1], self.rfoot_move[cur_idx-1:cur_idx+1])
                        if init_vel_loss:
                            if init_first_slide:
                                flsld[i] += init_vel_loss.item() * 100
                                sliding_num += 1
                                init_first_slide = False
                            # lsld[i] += init_vel_loss.item() * 100
                            sum_loss += self.args.init_vel_weight * init_vel_loss
                            print_str += f'Isld {init_vel_loss.item()*100:.4f} '

                    # if self.args.init_rot_weight:
                    #     if i!= 0:
                    #         init_rot = self.synced_mocap_rots_axis[cur_idx - 1, :3]
                    #         gt_matrs = R.from_rotvec(init_rot).as_matrix()
                    #         init_rot = torch.tensor(gt_matrs).type(torch.FloatTensor).cuda().unsqueeze(0)

                    #         seg_first_rot = orientations[0, 0, :, :].unsqueeze(0)
                    #         init_rot_loss = joint_orient_error(seg_first_rot, init_rot)
                            
                    #         sum_loss += self.args.init_rot_weight * init_rot_loss
                    #         print_str += f'Init rot {init_rot_loss.item():.3f} '

                if torch.any(torch.isnan(mocap_ori_params)):
                    print('mocap_ori_params nan')
                if torch.any(torch.isnan(mocap_after_ori_params)):
                    print('mocap_after_ori_params nan')
                if sum_loss > 0:
                    sum_loss.backward()
                    optimizer.step()
                time2 = time.time()
                
                if torch.any(torch.isnan(mocap_ori_params)):
                    print('mocap_ori_params nan')
                if torch.any(torch.isnan(mocap_after_ori_params)):
                    print('mocap_after_ori_params nan')

                mocap_rots = torch.cat([mocap_ori_params, mocap_after_ori_params], dim=1)

                # add scalar to sumwriter

                # print(f)
                if iters == 0:
                    print(print_str + f'Loss {sum_loss:.3f} Time {(time2 - time_start):.1f}s')
                else:
                    print('\r' + print_str + f'Loss {sum_loss:.3f} Time {(time2 - time_start):.1f}s', end = '', flush=True)
                if abs(pre_loss - sum_loss) < 1e-4 or iters + 1 == self.args.iterations :
                    pre_lidar_trans = lidar_trans_params[-1:].clone().detach()
                    pre_smpl_verts, _ = self.mocap_to_smpl(mocap_rots[-1:].clone().detach(), pre_lidar_trans)
                    # pre_state = foot_states[-1].copy()
                    pre_loss = sum_loss.item()
                    print('')
                    break

            loss_dict[i]['Sum Contact loss'] = lcont.sum() / contact_num
            loss_dict[i]['contact loss w/o opt'] = flcont.sum() / contact_num
            loss_dict[i]['Sliding loss'] = lsld.sum() / sliding_num
            loss_dict[i]['Sliding loss w/o opt'] = flsld.sum() / sliding_num
            loss_dict[i]['smooth loss'] = lsmt.sum() / trans_num
            loss_dict[i]['smooth loss w/o opt'] = flsmt.sum() / trans_num

            loss_dict[i]['Seg Contact loss'] = lcont[i] / cnum
            loss_dict[i]['Seg Sliding loss'] = lsld[i] / snum
            loss_dict[i]['sliding_num'] = sliding_num
            loss_dict[i]['contact_num'] = contact_num
            loss_dict[i]['trans_num'] = trans_num
            loss_dict[i]['time'] = time.time() - time_start

            self.writer.add_scalars('total/mean', {'contact': lcont.sum() / contact_num, 'sliding': lsld.sum() / sliding_num, 'trans_smt': lsmt.sum() / trans_num}, i)
            self.writer.add_scalars('total/mean', {'contact w/o': flcont.sum() / contact_num, 'sliding w/o': flsld.sum() / sliding_num, 'trans_smt w/o': flsmt.sum() / trans_num}, i)

            self.writer.add_scalars(
                'total/window', {'contact': lcont[i] / cnum, 'sliding': lsld[i] / snum, 'trans_smt': lsmt[i] / tnum}, i)
            self.writer.add_scalars(
                'total/window', {'contact': flcont[i], 'sliding': flsld[i], 'trans_smt': flsmt[i]}, i)

            print('================================================================')
            print('Contact loss: ', lcont.sum() / contact_num)
            print('Contact loss w/o opt: ', flcont.sum() / contact_num)
            print('Sliding loss: ', lsld.sum() / sliding_num)
            print('sliding loss w/o opt: ', flsld.sum() / sliding_num)
            print('smooth loss: ', lsmt.sum() / trans_num)
            print('smooth loss w/o opt: ', flsmt.sum() / trans_num)

            # 前一个优化窗口的偏移量，直接应用在下一个窗口上
            delta_trans = lidar_trans_params.detach().cpu().numpy() - self.synced_lidar[start:end, 1:4]
            delta_trans = delta_trans.mean(axis=0)
            print('delta_trans: ', delta_trans)
            print('================================================================')

            self.synced_lidar[start: end, 1:4] = lidar_trans_params.detach().cpu().numpy()

            os.makedirs(os.path.join(self.args.dataset_root, 'optimized'), exist_ok=True)
            
            save_json_file(os.path.join(self.args.dataset_root,
                           'optimized', self.save_str + '_loss.json'), loss_dict)
            save_traj_in_dir(os.path.join(self.args.dataset_root, 'optimized'),
                             self.synced_lidar[:end],  self.save_str + '_optimized')
            multiprocess_save_smpl_model(self.synced_lidar[start, 0], smpl_verts.detach().cpu().numpy(), self.args.dataset_root, self.args.data_name, 1)

        self.writer.close()
        print('Contact loss: ', lcont.sum() / contact_num)
        print('init contact loss: ', flcont.sum() / contact_num)
        print('Sliding loss: ', lsld.sum() / sliding_num)
        print('init sliding loss: ', flsld.sum() / sliding_num)


if __name__ == '__main__':
    
    from configs.config_loader import config_parser
    parser = config_parser(True)
    parser.add_argument("--save_smpl", type=bool, default = False)
    parser.add_argument("--cal_smpl_loss", type=bool, default = False)
    parser.add_argument("--new_traj", type=str,
                        default='', help='The lidar traj file path')
    parser.add_argument("-ST", "--save_step", type=int, default=1)
    parser.add_argument("-OS", "--opt_start", type=int, default=0)
    parser.add_argument("-OE", "--opt_end", type=int, default=-1)
    args = parser.parse_args()
    # exit()
    args.dataset_root = os.path.join(args.dataset_root, args.data_name) # 用于最新的数据集格式
    print('File path: ', args.dataset_root)
    print('File name: ', args.data_name)

    print(args)

    optimizer = Optimizer(args)
    print('==================================')
    if args.cal_smpl_loss or args.new_traj:
        print(f'Calculate smpl loss on {args.new_traj}...')
        print('==================================')
        optimizer.cal_mocap_loss(args.new_traj)
    elif args.save_smpl:
        print('save smpl')
        print('==================================')
        optimizer.save_smpl_models(args.save_step)
    else:
        print('Run optimization')
        print('==================================')
        optimizer.run()
