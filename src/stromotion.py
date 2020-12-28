import cv2
import copy
import numpy as np
#from jc_tools.jc import jfd
from yolact_src.eval import mian_init,evalimage,evalimagexia
from hp_sort.tools_bsp import BspFit,bsp_for_kp2d
from hp_sort.tools_op import load_npz,KpsToBboxes,op25b_sort_linspace
from hp_sort.tools_sift import Stitcher,match_distance,EnhanceImage,distance_line,WarpPerspectiveMatrix,WarpPerspectiveMatrix
PointDistance = 3000
path = 'yolact_src/weights/yolact_base_54_800000.pth'
net = mian_init(path)
#静止视角Stromotion
#增加iou阈值判断画出的动作
class StromotionCrossvideo():
    def __init__(self,workspace,path_2dpose,list_point,path_video,extract=10,show_foot_track_line=True,iou=0.2):
        self.list_point = list_point
        self.extract = extract
        self.show_foot_track_line = show_foot_track_line
        self.workspace = workspace
        self.path_2dpose = path_2dpose
        self.path_video = path_video
        self.stitcher = Stitcher()
        self.output = {'kp2d':{},'track':{}}
        self.foot = None
        self.iou = iou
        self.old_bbox=[]
        self.out = cv2.VideoWriter(self.workspace+'StromtionCrossVideo.mp4',cv2.VideoWriter_fourcc(*'XVID'), 5, (1920,1080),True)
        self.main()
    def map_srcimage_pan(self,src,dst,i): 
        vis = self.stitcher.stitch(i,[src, dst], ratio=0.75,showMatches=False,reprojThresh=10.0)
        #cv2.imshow('vis',cv2.resize(vis,None,fx=0.2,fy=0.2))
    def get_track(self):
        list_track,list_frames = [],[]
        pose1 = np.load(self.path_2dpose,allow_pickle=True )['op25b'].item()
        for k1,v1 in pose1.items():
            point = self.comput_distance_image(v1[0])
            if point!=False:
                list_track.append(point)
                list_frames.append(k1)
        return list_track,list_frames
    def get_bbox(self,start_frame,end_frame):
        list_bboxs,frame=[],[]
        pose1 = np.load(self.path_2dpose,allow_pickle=True )['op25b'].item()
        #pose2 = np.load('D:/Dancepose/BuildSrcImage/camera3_2dpose.npz',allow_pickle=True)['op25b'].item()
        for k1,v1 in pose1.items():
            frame.append(k1)
            list_bboxs.append(v1[0])
            
            #kp2ds.append(v1[0])
        np_bbox = np.array(list_bboxs)
        op2dpose = op25b_sort_linspace(np_bbox,frame,start_frame,end_frame)
        #print(np_bbox.shape)
        bboxs = KpsToBboxes(op2dpose)
        return bboxs,frame,list_bboxs
    def main(self):
        flag=False
        cap = cv2.VideoCapture(self.path_video)
        count = int(cap.get(7))
        start_frame,end_frame= 0,count
        bboxs,frames,kp2ds = self.get_bbox(start_frame,end_frame)  
        list_track,list_frames = self.get_track()
#         if self.list_point==[]:
#             list_track,list_frames = self.get_track()
#             foot_track,root_track = jfd(list_track,list_frames,self.path_2dpose)
#             self.list_point = list(foot_track)
        save_track=(int(list_track[0][0]),int(list_track[0][1]))
        np_track = np.array(list_track)
        np_track = np_track.reshape((np_track.shape[0],1,2))
        bsp_track = bsp_for_kp2d(np_track,alpha=0.0001)#BSP数据
        #print(len(frames),len(bboxs))
        list_bbox,list_borad_src,list_borad_dst,list_zero=[],[],[],[]
        for i in range(count):
            ret,src_im = cap.read()
            if ret and i in frames:
                if i%self.extract ==0 and len(self.list_point)==0:
                    flag= True
                if self.list_point!=[] and i in self.list_point or i%self.extract ==0:
                    flag= True
                if flag== True:
                    flag=False
                    one_bbox = bboxs[frames.index(i)].reshape((1,2,2))
                    #bbox_out = cv2.perspectiveTransform(one_bbox,H)
                    bbox_out = one_bbox.reshape(-1)
                    if len(self.old_bbox)!=0:
                        iou = self.cal_iou(bbox_out,self.old_bbox)
                        #print(iou)
                    else:
                        self.old_bbox=bbox_out
                        iou=0
                    if 0<iou<self.iou:
                       #print(dst_im.shape,src_im.shape)
                        #self.map_srcimage_pan(src_im,dst_im,i)
                        #H = self.stitcher.dict_H[i]
                        self.old_bbox=bbox_out
                        #src_im_trans = cv2.warpPerspective(dst_im,H,(dst_im.shape[1],dst_im.shape[0]))
                        bbox_im = src_im[int(bbox_out[1]):int(bbox_out[3]),int(bbox_out[0]):int(bbox_out[2])]
                        if bbox_im.shape[0]<=0:
                            continue
                        cv2.imshow('bbox_im',bbox_im)
                        cv2.waitKey(1)

                        fcn_im = copy.deepcopy(bbox_im)
                        #im_fcn = fcn_mask(fcn_im)
                        im_fcn = evalimagexia(net,fcn_im)
                        if isinstance(im_fcn,bool):
                            #print(i)
                            continue 
                        re = np.zeros(fcn_im.shape,dtype='uint8')
                        re[:,:,0]=re[:,:,1]=re[:,:,2]=im_fcn
                        re_src = re*fcn_im
                        #cv2.imshow('re',re_src)
                        #mask = 0 拥有背景
                        im_fcn = np.where(im_fcn[:,:]==[0],1,0)
                        re[:,:,0]=re[:,:,1]=re[:,:,2]=im_fcn
                        re_board = re*fcn_im
                        re_board = np.where(re_board[:,:]>[0],1,0)

                        #mask =0 背景全有的图片 
                        bbox_out = bbox_out.reshape(-1)
                        #dst_im_copy = copy.deepcopy(dst_im)
                        borad_dst = np.ones(src_im.shape,dtype='uint8')
                        borad_dst[int(bbox_out[1]):int(bbox_out[3]),int(bbox_out[0]):int(bbox_out[2])]=re_board
                        #dst_im_copy*=borad_dst

                        #mask =people 背景=0 
                        borad_src = np.zeros(src_im.shape,dtype='uint8')
                        borad_src[int(bbox_out[1]):int(bbox_out[3]),int(bbox_out[0]):int(bbox_out[2])]=re_src

                        list_borad_src.append(borad_src)
                        list_borad_dst.append(borad_dst)
                        self.output['kp2d'][frames.index(i)] = kp2ds[frames.index(i)]
                        im = copy.deepcopy(src_im)
                        for n in range(len(list_borad_src)-1):
                            borad_src = list_borad_src[n]
                            borad_dst = list_borad_dst[n]
                            im = im*borad_dst
                            im = im+borad_src
                        cv2.imshow('im',cv2.resize(im,None,fx=0.5,fy=0.5))
                        self.out.write(im)
                        cv2.waitKey(1)
                if i < bsp_track.shape[0]:
                    center = bsp_track[i]
                    list_zero.append((int(center[0]),int(center[1])))
                    save_track=(int(center[0]),int(center[1]))
                    if self.show_foot_track_line :
                        if len(list_zero)>1:
                            for num in range(len(list_zero)-1):
                                cv2.line(src_im,list_zero[num],list_zero[num+1],(0,0,255),3)
                self.output['track'][frames.index(i)] = save_track
        self.output['image']=im
        self.output['imagepath']=self.workspace+'StromtionCrossVideo.png'
        cv2.imwrite((self.workspace+'StromtionCrossVideo.png'),im)
        self.out.release()
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
    def comput_distance_image(self,pose,c=0.2):
            point_foot_right,point_foot_left = [22,23,24],[19,20,21]
            foot_left_compute,foot_right_compute=[],[]
            for i in point_foot_right:
                if 0 not in pose[i] and pose[i][2]>c:
                    foot_right_compute.append(pose[i])
            for i in point_foot_left:
                if 0 not in pose[i] and pose[i][2]>c:
                    foot_left_compute.append(pose[i])
            if foot_right_compute!=[] or foot_left_compute!=[]:
                pass
            else:
                return False
            #找到存在并可信的脚
            #取y值较大的脚
            if len(foot_right_compute)!=0 and len(foot_left_compute)!=0:
                np_foot_left_compute=np.array(foot_left_compute)
                np_foot_right_compute=np.array(foot_right_compute)
                y_left = np.max(np_foot_left_compute[:,1])
                y_right = np.max(np_foot_right_compute[:,1])
                x_left_mean,y_left_mean = np.mean(np_foot_left_compute[:,0]),np.mean(np_foot_left_compute[:,1])
                x_right_mean,y_right_mean = np.mean(np_foot_right_compute[:,0]),np.mean(np_foot_right_compute[:,1])
                if self.foot!=None:
                    distance_left = distance_line([[x_left_mean,y_left_mean],self.foot])
                    distance_right = distance_line([[x_right_mean,y_right_mean],self.foot])
                    #像素y值最小且距离符合
                    if max(y_left,y_right)==y_left and distance_left<PointDistance:
                        x,y = np.mean(np_foot_left_compute[:,0]),np.mean(np_foot_left_compute[:,1])
                    elif distance_right<PointDistance:
                        x,y = np.mean(np_foot_right_compute[:,0]),np.mean(np_foot_right_compute[:,1]) 
                    elif distance_right<distance_left:
                        x,y = np.mean(np_foot_right_compute[:,0]),np.mean(np_foot_right_compute[:,1]) 
                    else:
                        x,y = np.mean(np_foot_left_compute[:,0]),np.mean(np_foot_left_compute[:,1])
                else:
                    if max(y_left,y_right)==y_left:
                        x,y = np.mean(np_foot_left_compute[:,0]),np.mean(np_foot_left_compute[:,1])
                        self.foot=[x,y]
                    else:
                        x,y = np.mean(np_foot_right_compute[:,0]),np.mean(np_foot_right_compute[:,1])
                        self.foot=[x,y]
            else:
                if len(foot_right_compute)!=0:
                    np_foot_right_compute=np.array(foot_right_compute)
                    x,y = np.mean(np_foot_right_compute[:,0]),np.mean(np_foot_right_compute[:,1])
                else:
                    np_foot_left_compute=np.array(foot_left_compute)
                    x,y = np.mean(np_foot_left_compute[:,0]),np.mean(np_foot_left_compute[:,1])
            return (int(x),int(y))
    def cal_iou(self,box1,box2):
        """
        :param box1: = [xmin1, ymin1, xmax1, ymax1]
        :param box2: = [xmin2, ymin2, xmax2, ymax2]
        :return: 
        """
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        # 计算每个矩形的面积
        s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
        s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积
        # 计算相交矩形
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)
        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        area = w * h  # C∩G的面积
        iou = area / (s1 + s2 - area)
        return iou
