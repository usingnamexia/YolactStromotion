# YolactStromotion
在视频中人体完成2d关节点检测 及 bboxs后  
1 根据bboxs  切割画面人体  
2 Yolact实例分割 完成对靠前人体的mask输出   
3 根据叠加的mask 累积演示到最后的图片中  
4 输出stromotion 图片及形成的视频
## 视频演示:  
![image](https://github.com/usingnamexia/YolactStromotion/blob/main/demo/StromtionCrossVideo.gif)  
图片演示:  
![image](https://github.com/usingnamexia/YolactStromotion/blob/main/demo/StromtionCrossVideo_mini.png)  
## 使用介绍：  
S = StromotionCrossvideo(workspace,path_2dpose,path_video,extract=3)  
workspace：视频文件所在目录  
path_2dpose：op25b的npz文件路径  
path_video：视频路径  
extract：抽帧参数  
return：S.output  
格式：{'kp2d':{帧号：2dpose},'track':{帧号：（x,y）}}
在视频目录生成video 和 stromotionImage  
模型文件下载：  
链接：https://pan.baidu.com/s/1FMDH_WvzcBK8anz6mwnW4A   
提取码：ftf0   
放在yolact_src\weights\下即可  
