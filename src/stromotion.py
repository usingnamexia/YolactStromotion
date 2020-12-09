import cv2
import copy
import numpy as np
from yolact_src.eval import mian_init,evalimage,evalimagexia
from hp_sort.tools_op import load_npz,KpsToBboxes,op25b_sort_linspace
from hp_sort.tools_sift import Stitcher,match_distance,EnhanceImage,distance_line,WarpPerspectiveMatrix,WarpPerspectiveMatrix
path = 'yolact_src/weights/yolact_base_54_800000.pth'
net = mian_init(path)
#移动视角Stromotion
class StromotionCrossvideo():
    def __init__(self,workspace,path_2dpose,path_video,extract=10):
        self.extract = extract
        self.workspace = workspace
        self.path_2dpose = path_2dpose
        self.path_video = path_video
        self.stitcher = Stitcher()
        self.out = cv2.VideoWriter(self.workspace+'StromtionCrossVideo.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (1920,1080),True)
        self.main()
    def map_srcimage_pan(self,src,dst,i): 
        vis = self.stitcher.stitch(i,[src, dst], ratio=0.75,showMatches=False,reprojThresh=10.0)
        #cv2.imshow('vis',cv2.resize(vis,None,fx=0.2,fy=0.2))

    def get_bbox(self):
        list_bboxs,frame=[],[]
        pose1 = np.load(self.path_2dpose,allow_pickle=True )['op25b'].item()
        #pose2 = np.load('D:/Dancepose/BuildSrcImage/camera3_2dpose.npz',allow_pickle=True)['op25b'].item()
        for k1,v1 in pose1.items():
            frame.append(k1)
            list_bboxs.append(v1[0])
        np_bbox = np.array(list_bboxs)
        #print(np_bbox.shape)
        bboxs = KpsToBboxes(np_bbox)
        return bboxs,frame
    def main(self):
        cap = cv2.VideoCapture(self.path_video)
        count = int(cap.get(7))
        bboxs,frames = self.get_bbox()     
        print(len(frames),len(bboxs))
        list_bbox,list_borad_src,list_borad_dst=[],[],[]
        for i in range(count):
            ret,src_im = cap.read()
            if ret and i in frames:
                if i%self.extract ==0:
                   #print(dst_im.shape,src_im.shape)
                    #self.map_srcimage_pan(src_im,dst_im,i)
                    #H = self.stitcher.dict_H[i]
                    one_bbox = bboxs[frames.index(i)].reshape((1,2,2))
                    #bbox_out = cv2.perspectiveTransform(one_bbox,H)
                    bbox_out = one_bbox.reshape(-1)

                    #src_im_trans = cv2.warpPerspective(dst_im,H,(dst_im.shape[1],dst_im.shape[0]))
                    bbox_im = src_im[int(bbox_out[1]):int(bbox_out[3]),int(bbox_out[0]):int(bbox_out[2])]
                    cv2.imshow('bbox_im',bbox_im)
                    cv2.waitKey(1)
                    
                    fcn_im = copy.deepcopy(bbox_im)
                    #im_fcn = fcn_mask(fcn_im)
                    im_fcn = evalimagexia(net,fcn_im)
                    if isinstance(im_fcn,bool):
                        print(i)
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

                    im = copy.deepcopy(src_im)
                    for n in range(len(list_borad_src)-1):
                        borad_src = list_borad_src[n]
                        borad_dst = list_borad_dst[n]
                        im = im*borad_dst
                        im = im+borad_src
                    cv2.imshow('im',cv2.resize(im,None,fx=0.25,fy=0.25))
                    self.out.write(im)
                    cv2.waitKey(1)
        cv2.imwrite((self.workspace+'StromtionCrossVideo.png'),im)
        self.out.release()
        cv2.waitKey(1000)
        cv2.destroyAllWindows()