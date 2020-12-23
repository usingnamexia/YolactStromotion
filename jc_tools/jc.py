import cv2

#from jc_tools.jump_feature_detect import *
# from jump_feature_detect import map_op25b_to_h36m17
# from jump_feature_detect import plot_track
# from jump_feature_detect import calc_dist
# from jump_feature_detect import ploy_fit
# from jump_feature_detect import ploy_fit_all
# from jump_feature_detect import jump_keypoint_index
# from jump_feature_detect import polt_track_feature 
# from jump_feature_detect import detect_feature_area

from matplotlib import pyplot as plt
import numpy as np

from jc_tools.physical_tools import draw_canvas
from jc_tools.physical_tools import get_perspective_mat
from jc_tools.physical_tools import get_physical_quantity
def ploy_fit(x, y, deg):
    # np.polynomial.Polynomial.fit(x, y, deg)
    _poly = np.polyfit(x, y, deg)
    P = np.poly1d(_poly)
    return P

def ploy_fit_all(head_indexes, head_kps, root_indexes, root_kps, ploy_deg):
    Poly_h_yt = ploy_fit(head_indexes, head_kps[:, 1], ploy_deg)
    Poly_h_xt = ploy_fit(head_indexes, head_kps[:, 0], ploy_deg)
    Poly_r_yt = ploy_fit(root_indexes, root_kps[:, 1], ploy_deg)
    Poly_r_xt = ploy_fit(root_indexes, root_kps[:, 0], ploy_deg) 

    return Poly_h_yt, Poly_h_xt, Poly_r_yt, Poly_r_xt
def jump_keypoint_index(Ploy, frame_count):
    P_V = Ploy.deriv(m=1)
    P_Acc = Ploy.deriv(m=2)
    # V_0 = np.real(P_V.roots) 
    # Acc_0 = np.abs(P_Acc.roots)

    # V_0 = np.abs(P_V.roots)  # modulus of roots P_V(x) == 0 
    # V_0 = [np.real(var) for var in P_V.roots if np.abs(var) == np.imag(var) ]
    V_0 = [np.real(var) for var in P_V.roots if np.isreal(var) and np.real(var) > 0 and np.real(var) < frame_count]
    Acc_0 = [np.real(var) for var in P_Acc.roots if np.isreal(var) and np.real(var) > 0 and np.real(var) < frame_count]
    
    #print(f'V_0: {V_0}')
    #print(f'Acc_0: {Acc_0}')

    top_index = V_0[np.argmin(Ploy(V_0))]
    #print(f'top_index: {top_index}')

    Acc_jstop_index = np.where(Acc_0 > top_index)[0][-1]
    Acc_jstart_index = np.where(Acc_0 < top_index)[0][0]
    jstart_index = Acc_0[Acc_jstart_index]
    jstop_index = Acc_0[Acc_jstop_index]

    # print(f'jstart_index: {jstart_index}')
    # print(f'jstop_index: {jstop_index}')
    return int(jstart_index), int(top_index), int(jstop_index) 

def map_op25b_to_h36m17(kpts):
    assert kpts.shape[1]==25 and kpts.ndim==3
    select_index = [0,12,14,16,11,13,15,0,17,0,18,5,7,9,6,8,10]
    ret_kpts = kpts[:,select_index]
    
    ret_kpts[:,0] = (kpts[:,12]+kpts[:,11])/2
    ret_kpts[:,7] = (ret_kpts[:,8]+ret_kpts[:, 0])/2
    ret_kpts[:,9] = (ret_kpts[:,8]+ret_kpts[:,10])/2
    ret_kpts[:,0] = np.where(ret_kpts[:,1]==0, 0, ret_kpts[:,0])
    ret_kpts[:,0] = np.where(ret_kpts[:,4]==0, 0, ret_kpts[:,0])
    ret_kpts[:,7] = np.where(ret_kpts[:,0]==0, 0, ret_kpts[:,7])
    ret_kpts[:,7] = np.where(ret_kpts[:,8]==0, 0, ret_kpts[:,7])
    ret_kpts[:,9] = np.where(ret_kpts[:,8] ==0, 0, ret_kpts[:,9])
    ret_kpts[:,9] = np.where(ret_kpts[:,10]==0, 0, ret_kpts[:,9])
    return ret_kpts


# from jc_tools.numpy_tools import load_kp2ds_npz
from pathlib2 import Path
def load_kp2ds_npz(path, conf=False, fmt="op25b", key_joint=False, dict_fmt=False):
    data = np.load(path, allow_pickle=True)['op25b'].item()
    frame_indexes = np.array(list(data.keys())).squeeze()
    kp2ds = np.array(list(data.values())).squeeze()
    if fmt == "vp17":
        kp2ds = map_op25b_to_h36m17(kp2ds)
    if not (conf is False):
        if key_joint is False:
            conf_all = kp2ds[..., -1]
            # print(conf_all.shape, 1)
        else:
            conf_all = kp2ds[..., -1][:, key_joint]
            # print(conf_all.shape, 2)
        index_to_delete = np.unique(np.argwhere(conf_all<= conf)[:, 0])
        frame_indexes = np.delete(frame_indexes, index_to_delete, axis=0)
        kp2ds = np.delete(kp2ds, index_to_delete, axis=0)

    if dict_fmt == True:
        return dict(zip(frame_indexes, kp2ds))
    elif dict_fmt == False:
        return frame_indexes, kp2ds
    
def get_track(path_2dpose):
    #print(frame)
    list_track,list_frame=[],[]
    pose1 = np.load(path_2dpose,allow_pickle=True )['track'].item()
    for k,v in pose1.items():
        list_track.append(v)
        list_frame.append(k)
    #pose2 = np.load('D:/Dancepose/BuildSrcImage/camera3_2dpose.npz',allow_pickle=True)['op25b'].item()
    return list_track,list_frame

def jfd(list_track,list_frame,kp2ds_npz):
    frame_indexes, kp2ds = load_kp2ds_npz(kp2ds_npz, conf=0, fmt="vp17", key_joint=[0])
    #ead_kps_raw = kp2ds[:,8][:, :2]
    root_kps_raw = kp2ds[:,0][:, :2]
    np_track = np.array(list_track)
    ploy_deg = 16
    foot_kps,root_kps = np_track[:,:2],root_kps_raw
    foot_indexes,root_indexes = list_frame,frame_indexes
    Poly_h_yt, Poly_h_xt, Poly_r_yt, Poly_r_xt = ploy_fit_all(foot_indexes, foot_kps, root_indexes, root_kps, ploy_deg)
    foot_index_max = np.max(foot_indexes)
    root_index_max = np.max(root_indexes)
    jpk_fyt_indexes = jump_keypoint_index(Poly_h_yt, foot_index_max)
    jpk_ryt_indexes = jump_keypoint_index(Poly_r_yt, root_index_max)
   # print(f'jpk_hyt_indexes: {jpk_hyt_indexes}')
    #print(f'jpk_ryt_indexes: {jpk_ryt_indexes}')
    return jpk_fyt_indexes,jpk_ryt_indexes