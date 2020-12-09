import cv2
import os
import numpy as np
import json
import glob
import copy
import math

def OpJsonToNpz(fpath):
    jsons = glob.glob(f'{fpath}/*.json')
    data = {}
    for i in range(len(jsons)):
        print('\r%s'%(jsons[i]),end='')
        data[i] = LoadOpJson(jsons[i])
    np.savez(f'{os.path.dirname(fpath)}.npz',op25b=data)

def LoadOpJson(json_fpath):
    with open(json_fpath,'r') as f:
        json_data = json.load(f)
        N = len(json_data['people'])
        kp2ds = []
        if N == 0:
            return None
        for i in range(N):
            kp2ds.append(json_data['people'][i]['pose_keypoints_2d'])
        kp2ds = np.array(kp2ds).reshape((-1,25,3))
        return kp2ds
def KpsToBboxes(kps_):
    '''
    input: kps shape:Nx25x3
    outut: bbox shape:Nx4 ---> [Xmin,Ymin,Xmax,Ymax]
    '''
    kps = kps_.copy()
    bbox = []
    for kp in kps:
        x_max,y_max = max(kp[:,0]),max(kp[:,1])
        kp[:,0],kp[:,1] = np.where(kp[:,0]==0,x_max,kp[:,0]), np.where(kp[:,1]==0,y_max,kp[:,1])
        x_min,y_min = min(kp[:,0]),min(kp[:,1])
        bbox.append([x_min-20,y_min-20,x_max+20,y_max+20])
    bbox = np.array(bbox)
    return bbox
def KpsToBboxesCenter(kps_):
    '''
    input: kps shape:Nx25x3
    outut: bbox shape:Nx4 ---> [Xmin,Ymin,Xmax,Ymax]
    '''
    kps = kps_.copy()
    bbox = []
    for kp in kps:
        x_max,y_max = max(kp[:,0]),max(kp[:,1])
        kp[:,0],kp[:,1] = np.where(kp[:,0]==0,x_max,kp[:,0]), np.where(kp[:,1]==0,y_max,kp[:,1])
        x_min,y_min = min(kp[:,0]),min(kp[:,1])
        x_center,y_center = int((x_min+x_max)*0.5),int((y_min+y_max)*0.5)
        bbox.append([x_center,y_center])
    bbox = np.array(bbox)
    return bbox

def concat_npz(fpath_npz):
    a = []
    for i in glob.glob(f'{fpath_npz}/*.npz'):
        b = load_npz(i)
        a.append(b)
        os.remove(i)
    a = np.array(a).transpose((1,0,2,3))
    np.save(f'{fpath_npz}/{os.path.basename(fpath_npz[:-1])}.npy', a)


def load_npz(fpath_npz):
    data = np.load(fpath_npz, allow_pickle=True)['op25b'].item()
    N = 0
    for k in data.keys():
        if k > N:
            N = k
    kp2ds = np.zeros((N,25,3))
    for k in data.keys():
        kp2ds[k:k+1] = data[k][0]
    return kp2ds

def from_opkps_return_bbox_conf(kps):
    '''
    input: kps shape:Nx25x3
    outut: bbox shape:Nx4 ---> [Xmin,Ymin,Xmax,Ymax]
           ocnf Nx1
    ''' 
    bbox,conf,kp_index = [],[],[]
    index=0
    for kp in kps:
        if int(np.sum(kp[:,0])+np.sum(kp[:,1])) == 0:
            continue
        zero_num = max(sum(kp[:,0]==0),sum(kp[:,1]==0))
        conf.append(np.sum(kp[:,2])/(25-zero_num))
        kp = kp.astype(np.int)
        x_max,y_max = max(kp[:,0]),max(kp[:,1])
        kp[:,0] = np.where(kp[:,0]==0,x_max,kp[:,0])
        kp[:,1] = np.where(kp[:,1]==0,y_max,kp[:,1])
        x_min,y_min = min(kp[:,0]),min(kp[:,1])
        bbox.append([x_min,y_min,x_max,y_max])
        kp_index.append(index)
        index+=1
    if bbox is []:
        return None,None,None
    bbox,conf = np.array(bbox),np.array(conf)
    return bbox,conf,kp_index

def mc_load_json(json_fpath):
    with open(json_fpath,'r') as f:
        json_data = json.load(f)
        kp2ds = json_data['person']
        person_num = len(kp2ds)
        if person_num == 0:
            return None
        KP2DS,ID,BBOXS = [],[],[]
        for person_kp2ds in kp2ds:
            KP2DS.append(person_kp2ds['kp'])
            ID.append(person_kp2ds['id'])
            BBOXS.append(person_kp2ds['bbox'])
        return [KP2DS,ID,BBOXS]
    
def mc_save_json(im_name, im_ids, im_bboxs, im_kps, save_json_fpath):
    with open(save_json_fpath, 'w') as file:
        data = {'im_name':im_name,'person':[]}
        for id_,bbox,kp in zip(im_ids,im_bboxs,im_kps):
            data['person'].append({'id':id_,'bbox':bbox,'kp':kp})
        json.dump(data, file)

def load_mc_jsons(fpath):
    kp2ds,ids,bboxs = [],[],[]
    for i in glob.glob(f'{fpath}/*.json'):
        [a,b,c] = mc_load_json(i)
        kp2ds.append(a)
        ids.append(b)
        bboxs.append(c)
    kp2ds = np.array(kp2ds).squeeze()
    ids = np.array(ids).squeeze()
    bboxs = np.array(bboxs).squeeze()
    return kp2ds,ids,bboxs
        
def op25b_to_mc25b(opjson_fpath):
    im_list = glob.glob(f'{opjson_fpath[:-6]}/*.jpg')
    json_save_fpath = f'{opjson_fpath[:-6]}_mc25/'
    if os.path.isdir(json_save_fpath) is False:
        os.mkdir(json_save_fpath)
    kp = load_op(opjson_fpath)
    bb = from_opkps_return_bbox(kp)
    for i,j,k in zip(kp,bb,im_list):
        print('\r %s'%(k),end='')
        im_name = os.path.basename(k)
        im_ids = [0]
        i,j = i.reshape((-1,25,3)),j.reshape((-1,4))
        save_name = im_name.replace('jpg','json')
        mc_save_json(im_name, im_ids, j.tolist(), i.tolist(), f'{json_save_fpath}/{save_name}')
        
def map_op25b_to_h36m17(kpts):
    assert kpts.shape[1]==25 and kpts.ndim==3
    select_index = [0,12,14,16,11,13,15,0,17,0,18,5,7,9,6,8,10]
    ret_kpts = kpts[:,select_index]
    ret_kpts[:,0] = (kpts[:,12]+kpts[:,11])/2
    ret_kpts[:,7] = (ret_kpts[:,8]+ret_kpts[:, 0])/2
    ret_kpts[:,9] = (ret_kpts[:,8]+ret_kpts[:,10])/2
    return ret_kpts

def LoadR8Npz(fpath_npz, l, N):
    '''
    l:正方形的边长,N序列长度
    读取R8视频生成的npz,把2d点旋转到原始图片的位置
    '''
    data = np.load(fpath_npz, allow_pickle=True)['op25b'].item()  
    kp2ds = np.zeros((N,25,3))
    for k in data.keys():
        kp2ds[k:k+1] = data[k][0]
    kp2ds_result = kp2ds[::8].copy()
    points_center = np.zeros((25,2))+l/2
    kp2dsL8Copy_mask = np.ones((25,2))
    offset = int((l-math.sqrt(l**2/2))/2)
    for i in range(int(N/8)):
        kp2dsL8 = kp2ds[i*8:(i+1)*8]
        idx_cfds = [0,0]
        for j in range(8):
            if np.sum(kp2dsL8[j,:,-1])>idx_cfds[1]:
                idx_cfds = [j, np.sum(kp2dsL8[j,:,-1])]
        kp2dsL80 = kp2dsL8[idx_cfds[0]]
        kp2dsL8Copy_mask[:,0],kp2dsL8Copy_mask[:,1] = np.where(kp2dsL80[:,0]==[0], 0, 1), np.where(kp2dsL80[:,1]==[0], 0, 1)
        kp2dsL8Copy = PointRotate(45*idx_cfds[0], kp2dsL8[idx_cfds[0],:,0:2].copy(), points_center)-offset
        kp2ds_result[i,:,0:2] = kp2dsL8Copy*kp2dsL8Copy_mask
        kp2ds_result[i,:,-1] = kp2dsL8[idx_cfds[0],:,-1].copy()
    return kp2ds_result

def LoadR8Npz1(fpath_npz, l, N):
    '''
    l:正方形的边长,N:序列长度
    读取R8视频生成的npz,保存原始的2d点,以及需要旋转的度数
    '''
    data = np.load(fpath_npz, allow_pickle=True)['op25b'].item()  
    kp2ds = np.zeros((N,25,3))
    for k in data.keys():
        kp2ds[k:k+1] = data[k][0]
    kp2ds_result = kp2ds[::8].copy()
    kp2ds_rotate = np.zeros((kp2ds_result.shape[0]))
    points_center = np.zeros((25,2))+l/2
    kp2dsL8Copy_mask = np.ones((25,2))
    offset = int((l-math.sqrt(l**2/2))/2)
    for i in range(int(N/8)):
        kp2dsL8 = kp2ds[i*8:(i+1)*8]
        idx_cfds = [0,0]
        for j in range(8):
            if np.sum(kp2dsL8[j,:,-1])>idx_cfds[1]:
                idx_cfds = [j, np.sum(kp2dsL8[j,:,-1])]
        kp2dsL80 = kp2dsL8[idx_cfds[0]]
        kp2ds_rotate[i] = idx_cfds[0]
        kp2dsL8[idx_cfds[0],:,0:2].copy()
        kp2dsL8Copy = kp2dsL8[idx_cfds[0],:,0:2].copy()
        kp2ds_result[i,:,0:2] = kp2dsL8Copy
        kp2ds_result[i,:,-1] = kp2dsL8[idx_cfds[0],:,-1].copy()
    return kp2ds_result,kp2ds_rotate*45

def KpsAddOffset(kp2ds, offset):
    N = kp2ds.shape[0]
    kp2ds_mask = np.ones((25,2))
    for i in range(N):
        kp2ds_mask[:,0],kp2ds_mask[:,1] = np.where(kp2ds[i,:,0]==[0], 0, 1), np.where(kp2ds[i,:,1]==[0], 0, 1)
        kp2ds[i]+=offset[i]
        kp2ds[i]*=kp2ds_mask
    return kp2ds
#pose 修复
 #插值补齐0值的pose
def op25b_sort_linspace(pose2d,list_frame,start_frame,end_frame):
    #print(list_frame)
    old = 0
    kp2d_notebook = {}
    '''pose2d shape is n*25*3'''
    for i in range(pose2d.shape[0]):
        pose=np.asarray([pose2d[i]])
        b = np.where(pose==[0,0,0], 0, 1)[0,:,0]
        if np.sum(b)>=20:
            kp2d_notebook[i]=pose2d[i]
        else:
            print('\r %d'%i,end='')
    list_notebook = list(kp2d_notebook.keys())
    list_linspace = []
    for i in range(len(list_notebook)):
        if i<len(list_notebook)-1:
            if list_notebook[i]+1 != list_notebook[i+1]:
                list_linspace.append([list_notebook[i],list_notebook[i+1]])
    for loss in list_linspace:
        start_loss,end_loss = loss
        line_loss = np.linspace(pose2d[start_loss],pose2d[end_loss],end_loss-start_loss)
        for i in range(25):
            if 0 not in  pose2d[start_loss][i] and 0 not in  pose2d[end_loss][i]:
                pose2d[start_loss:end_loss,i]=line_loss[:,i]
    #+插值补齐未识别的帧
    len_pose = end_frame-start_frame
    add_2dpse = np.zeros((len_pose,25,3))
    list_linspace = []
    for i in range(len(list_frame)-1):
        if list_frame[i]+1 != list_frame[i+1]:
            list_linspace.append([list_frame[i],list_frame[i+1]])
        else:
            #print(i)
            add_2dpse[list_frame[i]-start_frame]=pose2d[i]
    print(list_linspace)
    for loss in list_linspace:
        start_loss,end_loss = loss
        line_loss = np.linspace(pose2d[list_frame.index(start_loss),:,0:2],pose2d[list_frame.index(end_loss),:,0:2],end_loss-start_loss)

        add_2dpse[(start_loss-start_frame):int(end_loss-start_frame),:,0:2]=line_loss
     
        add_2dpse[(start_loss-start_frame):int(end_loss-start_frame),:,2]=pose2d[list_frame.index(start_loss):list_frame.index(start_loss)+(end_loss-start_loss),:,2]
        print([start_loss-start_frame,end_loss-start_frame])
        
    return add_2dpse