# -*- coding: utf-8 -*-
"""
name:xia
time:2020/10/27
"""
#import tqdm
import numpy as np
import imutils
import cv2  
import time
import math

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3()
        self.dict_H={}
    def stitch(self, i,images, ratio=0.7, reprojThresh=4.0,
        showMatches=False):
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
        (matches, H, status) = M
        self.dict_H[i]=H
        # check to see if the keypoint matches should be visualized
        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)
            end = time.time()
            #print('%.5f s' %(end-start))
            # return a tuple of the stitched image and the
            # visualization
            return vis

        # return the stitched image
        return imageA
    #接收照片，检测关键点和提取局部不变特征
    #用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    #detectAndCompute方法用来处理提取关键点和特征
    #返回一系列的关键点
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
            #orb = cv2.ORB_create()
            #kp0,features = orb.detectAndCompute(gray,None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)
    #matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量。
    #David Lowe’s ratio测试变量和RANSAC重投影门限也应该被提供。
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None
    #连线画出两幅图的匹配
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
class StitcherSrc:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()
        self.dict_H={}
    def change_size(self,image):
        #image=cv2.imread(read_file,1) #读取图片 image_name应该是变量
        img = cv2.medianBlur(image,5) #中值滤波，去除黑色边际中可能含有的噪声干扰
        b=cv2.threshold(img,15,255,cv2.THRESH_BINARY)          #调整裁剪效果
        binary_image=b[1]               #二值图--具有三通道
        binary_image=cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
        #print(binary_image.shape)       #改为单通道
     
        x=binary_image.shape[0]
        #print("高度x=",x)
        y=binary_image.shape[1]
        #print("宽度y=",y)
        edges_x=[]
        edges_y=[]
        for i in range(x):
            for j in range(y):
                if binary_image[i][j]==255:
                    edges_x.append(i)
                    edges_y.append(j)
     
        left=min(edges_x)               #左边界
        right=max(edges_x)              #右边界
        width=right-left                #宽度
        bottom=min(edges_y)             #底部
        top=max(edges_y)                #顶部
        height=top-bottom               #高度
     
        pre1_picture=image[left:left+width,bottom:bottom+height]        #图片截取
        return pre1_picture                                             #返回图片数据

    def stitch(self, i,images, ratio=0.75, reprojThresh=4.0,
        showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        
        (imageB, imageA) = images
        start = time.time()
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        end = time.time()
        #print('%.5f s' %(end-start))

        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        start = time.time()
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)
        end = time.time()
        #print('%.5f s' %(end-start))


        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        self.dict_H[i]=H
        start = time.time()
        result = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        result = self.change_size(result)
        end = time.time()
        #print('%.5f s' %(end-start))


        # check to see if the keypoint matches should be visualized
        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)
            end = time.time()
            #print('%.5f s' %(end-start))
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result
    
    #接收照片，检测关键点和提取局部不变特征
    #用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    #detectAndCompute方法用来处理提取关键点和特征
    #返回一系列的关键点
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)
    #matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量。
    #David Lowe’s ratio测试变量和RANSAC重投影门限也应该被提供。
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None
    #连线画出两幅图的匹配
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
def match_distance(list_):
    #print(list_)
    d_x = list_[0][0]-list_[len(list_)-1][0]
    d_x = d_x*bili_C[0]
    d_y = list_[0][1]-list_[len(list_)-1][1]
    d_y = d_y*bili_C[1]
    distance = math.sqrt(d_x**2 + d_y**2)
    point0 = (int(list_[0][0]),int(list_[0][1]))
    point1 = (int(list_[len(list_)-1][0]),int(list_[len(list_)-1][1]))
    return point0,point1,distance
def distance_line(list_xy):
    p1=np.array(list_xy[0])
    p2=np.array(list_xy[1])
    p3=p2-p1
    p4=math.hypot(p3[0],p3[1])
    return int(p4)
def Op25bToTraj2d(kp2ds1):
    traj1_11,traj1_12 = kp2ds1[:,11].copy(),kp2ds1[:,12].copy()
    #traj2_11,traj2_12 = kp2ds2[:,11].copy(),kp2ds2[:,12].copy()
    traj1_11,traj1_12 = Complete2DPoints(traj1_11,0.2),Complete2DPoints(traj1_12,0.2)
    #traj2_11,traj2_12 = Complete2DPoints(traj2_11,0.2),Complete2DPoints(traj2_12,0.2)
    traj1= (traj1_11+traj1_12)/2
    return traj1
def Complete2DPoints(traj, threshold=0.1):
    '''points shape: Nx3 x,y,c'''
    N = traj.shape[0]
    lenth = 0
    for i in range(0,N-1,1):
        A,B = np.sum(abs(traj[i])),np.sum(abs(traj[i+1]))
        if A==0 and B==0:
            lenth+=1
            if i==N-2:
                traj[-lenth-1:] = traj[-lenth-2]
        if A==0 and B!=0:
            data = np.linspace(traj[i-lenth-1], traj[i+1], lenth+3)
            if data.shape[0]==traj[i-lenth-1:i+2].shape[0]:
                traj[i-lenth-1:i+2]=data
            else:
                traj[:i+1] = traj[i+1]
            lenth=0
    if np.sum(traj[-1])==0:
        for i in range(N-1,0,-1):
            if np.sum(traj[i])!=0:
                traj[i:] = traj[i]
                break
    return traj
#直方图均值化
def EnhanceImage(img):
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return result
def my_rotate(image,center,angle,color):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),borderValue=color)

def im_rotate(im,center,degree,color):
    h,w,_ = im.shape
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    print(M.shape)
    im_rotate = cv2.warpAffine(im, M, (h, w),borderValue=color)
    return im_rotate
record_point=[]

def check_enter(point,distance_flag):#[x,y]
    global record_point
    if len(record_point)==0:
        record_point = point
        return record_point
    else:
        #print(point,record_point)
        x1,y1 = point
        x0,y0 = record_point
        distance_real = abs((abs((y1-y0))**2)-(abs((x1-x0))**2)**0.5)
        #print(distance_real)
        if distance_real > distance_flag:
            pass
            #record_point=point
            #return record_point
        else:
            record_point=point
        return record_point
        
def get_kp2ds(frame,path_2dpose):
    #print(frame)
    list_kp2ds,list_frame=[],[]
    pose1 = np.load(path_2dpose,allow_pickle=True )['op25b'].item()
    for k,v in pose1.items():
        if k in frame:
            list_kp2ds.append(v)
            list_frame.append(k)
    #pose2 = np.load('D:/Dancepose/BuildSrcImage/camera3_2dpose.npz',allow_pickle=True)['op25b'].item()
    return list_kp2ds,list_frame

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = [ A_i[0], A_i[1], 1, 
                             0,      0, 0,
                       -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :]   = [      0,      0, 0,
                        A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 
    A = np.mat(A)
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    
    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix
 # 生成子弹时间的原图
    def buildBulletTime_origin(self, thresh_iou = 0.2,blend_sz = 3):
        '''
        input=》
            videoPath：视频路径
            dets：骨骼点数据
            draw_ts：要绘制的帧数合集
            frame_start：预加帧数
            thresh_iou：IOU比对系数（用于淡化）
        output=》
            bulletTime：生成的时间线原图
            ts_ret：最终相对于视频插入的真实的帧数，主要是方便后期需要
        '''
        frame_start = self.framestart
        ts_ret = []
    #     读取视频
        cap = cv2.VideoCapture(self.video_path)
    #     构建高斯混合模型
        knn2 = cv2.createBackgroundSubtractorKNN(history = 60, detectShadows=False)
        mog2 = cv2.createBackgroundSubtractorMOG2(history = 60, detectShadows=False)

        bulletTime = None
    #     从特定位置启动视频，此处先为0
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    #     获取到视频的总帧数
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     哪怕前面的不用，也进行建模
        for i in tqdm(range(frames if frame_start == 0 else frame_start)):
    #         cap.read()按帧读取视频，good,frame是获cap.read()方法的两个返回值。其中good是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
            good, frame = cap.read()    
            if not good or i>self.select_ts[-1] :
                break
            # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
            fg = cv2.bitwise_or(mog2.apply(frame),knn2.apply(frame))

        lastmask = None
    #     enumerate()返回下标和数据
        for i,ts in tqdm(enumerate(self.select_ts)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,ts+frame_start)
            good, frame = cap.read()    
            if not good or ts>self.h36m17.shape[0]:
                break

            fg = cv2.bitwise_or(mog2.apply(frame),knn2.apply(frame))
    #         图像阈值处理cv2.threshold()函数，cv2.THRESH_BINARY这个参数表是表示大于阈值的是最大值，小于时为零，这里得到的是二值图即黑白图
            mask = cv2.threshold(fg,0,255,cv2.THRESH_BINARY)[1]
    #     cv2.findContours()函数用来检测轮廓，且输入的必须是二值图
    #        第一个参数是图像
    #        第二个参数是检索方式 这里选择不建立等级
    #        第三个参数是轮廓的近似方法 这里选择存储全部轮廓点
    #    返回轮廓和每个轮廓的信息
            c,h = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    #     cv2.contourArea()函数用来求轮廓面积，这里将轮廓的numpy中的ndarray根据面积去排序
            c = sorted(c,key = lambda x:cv2.contourArea(x))
            mask *= 0
            [cv2.drawContours(mask, c,i,(255),-1,16) for i in range(len(c)) if cv2.contourArea(c[i])>5000 and cv2.boundingRect(c[i])[2]/cv2.boundingRect(c[i])[3]<self.boundingRectScale]

            iou = 0
            if lastmask is None:
                lastmask = mask
            else:
                iou = np.sum(cv2.bitwise_and(lastmask,mask))/np.sum(cv2.bitwise_or(lastmask,mask))
                if iou > thresh_iou:
                    continue
                print(iou)
            # 保存最终选定的帧号
            ts_ret.append(ts)
    #         ts_ret.append(ts+frame_start)
            lastmask = mask
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(blend_sz*2|1,blend_sz*2|1)))
            mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(blend_sz*4|1,blend_sz*2|1)))


            mask2 = cv2.ximgproc.guidedFilter(frame,mask,blend_sz,3)
            cv2.imshow("guidedFilter",mask2)
            mask = cv2.normalize(mask2.astype(np.float32),mask,0,1,cv2.NORM_MINMAX)
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            if iou > thresh_iou:
                mask*=(1-iou)/2
            bulletTime = cv2.addWeighted(cv2.multiply(frame.astype(np.float32),mask),1,
                                         cv2.multiply(bulletTime.astype(np.float32),1.-mask),1,
                                         0) if bulletTime is not None else frame

            cv2.imshow("bulletTime",bulletTime/255)
            cv2.imshow("mask",mask)
            if cv2.waitKey(1) == 27:
                break
        self.bulletTime_origin = bulletTime
        self.bulletTime_origin_path = self.outputdir+'/'+str(Path(self.video_path).stem)+"_bulletTime_origin"+".png"
        self.ts_final = ts_ret
        self.ts_final_txt_path = self.outputdir+'/'+str(Path(self.video_path).stem)+'_res.txt'
        cv2.imwrite(self.bulletTime_origin_path,bulletTime)
        np.savetxt(self.ts_final_txt_path,[t+self.framestart for t in self.ts_final],fmt='%i')
        return bulletTime,ts_ret
    