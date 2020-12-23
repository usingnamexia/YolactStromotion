#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# @File: e:\JC\Repository\JC_Figure_Skating\Track\physical_tools.py
# @Project: e:\JC\Repository\JC_Figure_Skating\Track
# @Created Date: Friday, June 12th 2020, 18:02:29
# @Author: Guo Yumen
# @Last Modified: Thursday, October 29th 2020, 16:21:41
# @Modified By: JC
# @Version: v2
# @Host Name:MC-AI-PC
###

import numpy as np #1
import cv2 #2
import math

def draw_canvas():
    # 1:25的比例尺
    # canvas = np.zeros((1200, 2118, 3), dtype="float32") # 高度+800 宽度+1000
    centerX = 1559 #竖直中间线横坐标
    centerY = 1000 #水平中间线纵坐标
    left = 500
    right = 2618
    left1 = 1204
    right1 = 1914
    top = 400
    bottom = 1600
    # canvas = np.zeros((2000, 3118, 3), dtype="float32")  # 3
    canvas = np.ones((2000, 3118, 3), dtype="uint8")*255  # 3
    
    blue = (255, 0, 0)  # 4
    red = (0, 0, 255)  # 8
    white = (255, 255, 255)
    cv2.line(canvas, (left1, top), (left1, bottom), blue)  # 5
    cv2.line(canvas, (right1, top), (right1, bottom), blue)  # 5
    cv2.line(canvas, (centerX, top), (centerX, bottom), red)  # 中间线
    # 左右边界
    cv2.line(canvas, (left, top), (left, bottom), red)
    cv2.line(canvas, (right, top), (right, bottom), red)

    cv2.line(canvas, (left, centerY), (right, centerY), white)
    # 上下边界
    cv2.line(canvas, (left, top), (right, top), red)
    cv2.line(canvas, (left, bottom), (right, bottom), red)

    # cv2.imshow("Canvas", canvas) #6
    cv2.imwrite("canvas01.png", canvas)
    #
    # cv2.waitKey(0)  # 7
    return canvas


def get_perspective_mat(src_points, dst_points):
    '''
    :param src_points:视频帧截图四个顶点坐标（点数组）
    :param dst_points:目标图像上四个顶点的坐标（点数组）
    :return: 透视矩阵
    '''
    # src_points = [[1073., 126.], [1430., 129.], [15., 1057.], [1221., 1240.]]
    src_points = np.array(src_points, dtype="float32")
    # dst_points = [[1559., 400.], [1914., 400.], [1559., 1600.], [1914., 1600.]]
    dst_points = np.array(dst_points, dtype="float32")

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # print(M)
    return M

def get_physical_quantity(start_point, end_point, M, jkp_indexes, fps=60, g=9.8):
    '''
    :param start_point: 起跳点像素坐标
    :param end_point: 落地点像素坐标
    :param total_time: 腾空时间帧差
    :param up_time: 上升时间帧数
    :return: 实际高度h, 初速度v0, 水平位移distance, 初速度与地面夹角angle, 水平初速度v0_x, 竖直初速度v0_y
    '''    
    # 对坐标进行透视变换
    # 起点[855, 507] 落地点[1044, 508], 冰刀尖

    # x1 = (start_point[0] * M[0][0] + start_point[1] * M[0][1] + M[0][2]) / (start_point[0] * M[2][0] + start_point[1] * M[2][1] + M[2][2])
    # y1 = (start_point[0] * M[1][0] + start_point[1] * M[1][1] + M[1][2]) / (start_point[0] * M[2][0] + start_point[1] * M[2][1] + M[2][2])
    # x2 = (end_point[0] * M[0][0] + end_point[1] * M[0][1] + M[0][2]) / (end_point[0] * M[2][0] + end_point[1] * M[2][1] + M[2][2])
    # y2 = (end_point[0] * M[1][0] + end_point[1] * M[1][1] + M[0][2]) / (end_point[0] * M[2][0] + end_point[1] * M[2][1] + M[2][2])
    # jump = [x1, y1]
    # land = [x2, y2]
    start_point = np.array(start_point).reshape(1, -1, 2)
    end_point = np.array(end_point).reshape(1, -1, 2) 
    
    jump = cv2.perspectiveTransform(start_point, M)
    land = cv2.perspectiveTransform(end_point, M)
    
    jump = jump.reshape(-1)
    land = land.reshape(-1)
     
    print("jump: ", jump)
    print("land: ", land)
    # 换算成米求勾股
    x = (land[0] - jump[0]) * 25 / 1000
    y = (land[1] - jump[1]) * 25 / 1000
    distance = math.sqrt(x * x + y * y)
    print(f'distance: ', distance)
    # 根据时间帧与秒的换算关系计算实际时间
    t = (jkp_indexes[-1] - jkp_indexes[0]) / fps
    # 通过X= Cosθ*v0*t，得到速度的水平分量x*Cosθ
    v0_x = distance / t

    # 根据像素坐标与人物实际身高，求得实际长度与px间的比例尺
    # scale = 1.62 / math.sqrt(38 * 38 + 243 * 243)
    
    t1 = (jkp_indexes[-2] - jkp_indexes[0]) / fps
    # 最高点腾空高度
    # h = h_px * scale
    # y=Sinθ*v0*t-1/2gt² g=9.8  ,可求得初速度的竖直分量
    h = 1/2 * g * (t/2)**2
    # v0_y = (h + 0.5 * g * t1 * t1) / t1
    v0_y = g * (t/2)

    # 求出初速度v0
    v0 = math.sqrt(v0_x * v0_x + v0_y * v0_y)

    # 求出初速度与地面夹角
    angle = math.atan(v0_y / v0_x)
    angle = math.degrees(angle)

    # print("最大高度= {}m"
    #       "，v0 = {}m/s"
    #       "，水平位移 = {}m， "
    #       ", 夹角 = {}°，"
    #       "水平速度 = {}m/s， 竖直速度 = {}m/s".format(h, v0, distance, angle, v0_x, v0_y))
    # return h, v0, distance, angle
    return v0, angle, h, distance

if __name__ == "__main__":
    canvas = draw_canvas()
    cv2.imshow("canvas", cv2.pyrDown(canvas))
    # cv2.waitKey(0)
# [ [ 835  368]
#   [ 930  348]
#   [1013  364]]
# [55 66 76]
    
    src_points = [[1073., 126.], [1430., 129.], [15., 1057.], [1221., 1240.]]
    dst_points = [[1559., 400.], [1914., 400.], [1559., 1600.], [1914., 1600.]]
    M = get_perspective_mat(src_points, dst_points)
    jkp_indexes = [55, 66, 76]
    v0, angle, h, distance = get_physical_quantity([855, 507],[1044, 508], M, jkp_indexes)

    print(canvas.size)
    size = canvas.shape[1], canvas.shape[0]
    image = cv2.warpPerspective(canvas, M, size)   
    cv2.imshow("image", cv2.pyrDown(image))
    cv2.waitKey(0)

    print(f'v0: {v0:.2f}m/s')
    print(f'angle: {angle:.2f}')
    print(f'h: {h:.2f}m')
    print(f'distance: {distance:.2f}m')

    cv2.destroyAllWindows()
