from ast import Break
import cv2 as cv
import random as rng
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import queue,easygui
import argparse
import traceback
def filter(img,HSV=((0,0,0),(360,255,255)),k_erode=3,k_dilate=5,erode_time=1,dilate_time=2):
    global low_H,high_H,low_S,high_S,low_V,high_V,k_size,k_dilatesize,erode_times,dilate_times
    low_H,low_S,low_V=HSV[0]
    high_H,high_S,high_V=HSV[1]
    k_size=k_erode
    k_dilatesize=k_dilate
    erode_times=erode_time
    dilate_times=dilate_time
    window_capture_name = 'Video Capture'
    window_detection_name = 'Object Detection'
    low_H_name = 'Low H'
    low_S_name = 'Low S'
    low_V_name = 'Low V'
    high_H_name = 'High H'
    high_S_name = 'High S'
    high_V_name = 'High V'
    def on_low_H_thresh_trackbar(val):
        global low_H
        global high_H
        low_H = val
        low_H = min(high_H-1, low_H)
        cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
    def on_high_H_thresh_trackbar(val):
        global low_H
        global high_H
        high_H = val
        high_H = max(high_H, low_H+1)
        cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
    def on_low_S_thresh_trackbar(val):
        global low_S
        global high_S
        low_S = val
        low_S = min(high_S-1, low_S)
        cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
    def on_high_S_thresh_trackbar(val):
        global low_S
        global high_S
        high_S = val
        high_S = max(high_S, low_S+1)
        cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
    def on_low_V_thresh_trackbar(val):
        global low_V
        global high_V
        low_V = val
        low_V = min(high_V-1, low_V)
        cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
    def on_high_V_thresh_trackbar(val):
        global low_V
        global high_V
        high_V = val
        high_V = max(high_V, low_V+1)
        cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
    def on_k_size_thresh_trackbar(val):
        global k_size
        if val%2 == 1:
            k_size = val
    def on_kdilate_size_thresh_trackbar(val):
        global k_dilatesize
        if val%2 == 1:
            k_dilatesize = val
    def on_erode_times_thresh_trackbar(val):
        global erode_times
        erode_times = val
    def on_dilate_times_thresh_trackbar(val):
        global dilate_times
        dilate_times = val
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
    cv.namedWindow(window_capture_name,cv.WINDOW_NORMAL)
    cv.namedWindow(window_detection_name,cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_capture_name,cv.WND_PROP_TOPMOST,1)
    cv.setWindowProperty(window_detection_name,cv.WND_PROP_TOPMOST,1)
    cv.createTrackbar(low_H_name, window_detection_name , low_H, 360, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name , high_H, 360, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name , low_S, 255, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name , high_S, 255, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name , low_V, 255, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name , high_V, 255, on_high_V_thresh_trackbar)

    cv.createTrackbar("k_size", window_detection_name , k_erode, 15, on_k_size_thresh_trackbar)
    cv.createTrackbar("dilate_k_size", window_detection_name , k_dilate, 15, on_kdilate_size_thresh_trackbar)
    cv.createTrackbar("erodetimes", window_detection_name , erode_time, 10, on_erode_times_thresh_trackbar)
    cv.createTrackbar("dilatetimes", window_detection_name , dilate_time, 14, on_dilate_times_thresh_trackbar)
    frame = img
    while True:
        if frame is None:
            break
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        k=cv.getStructuringElement(cv.MORPH_ELLIPSE,(k_size,k_size))
        k_dilate=cv.getStructuringElement(cv.MORPH_ELLIPSE,(k_dilatesize,k_dilatesize))
        frame_threshold = cv.erode(frame_threshold,k,iterations=erode_times)
        frame_threshold = cv.dilate(frame_threshold,k_dilate,iterations=dilate_times)
        
        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, frame_threshold)
        
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            break
    cv.destroyAllWindows()
    return (((low_H,low_S,low_V),(high_H,high_S,high_V)),(k_size,k_size),(k_dilatesize,k_dilatesize),erode_times,dilate_times)
class experimentVideo:
    def __init__(self,videoname,size=None):
        self.videoname=videoname
        self.size=size
        self.HSV=((0, 0, 0), (255, 255, 70))
        self.k_size=(5,5)
        self.k_dilate_size=(7,7)
        self.dilate_times=8
        self.erode_times=3
    def process(self,img,output=False):
        k=cv.getStructuringElement(cv.MORPH_ELLIPSE,self.k_size)
        kdilate=cv.getStructuringElement(cv.MORPH_ELLIPSE,self.k_dilate_size)
        img = cv.inRange(img,self.HSV[0], self.HSV[1])
        img = cv.erode(img,k,iterations=self.erode_times)
        img = cv.dilate(img,kdilate,iterations=self.dilate_times)
        if output:
            cv.namedWindow("output")
            cv.setWindowProperty("output",cv.WND_PROP_TOPMOST,1)
            cv.imshow("output",img)
        return img
    def getRect(self):
        video=cv.VideoCapture(self.videoname)
        fps=video.get(cv.CAP_PROP_FPS)
        plt.figure()
        rng.seed(12345)
        def thresh_callback(output):
            global last_vector
            global last_update
            global last_delta_fai
            global last_delta_fai_update
            global threshold
            global a_list
            global xLine
            global yLine
            global a_time
            global fai_xLine
            global fai_yLine
            contours, hierarchy = cv.findContours(output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # Draw contours
            drawing = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
            area=[]
            if len(contours) >= 2:
                for contour in contours:
                    area.append(cv.contourArea(contour)+rng.random())#加随机数是因为如果有俩面积大小一样，排序那步就会报错，也可以+index(contour)/100之类的？
                # print(len(contours))
                oricon=contours[:]
                res= zip(area,oricon)
                res=sorted(res,reverse=True)
                _,contours=zip(*res)
                x=[0,0]
                y=[0,0]
                for i in range(2):
                    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                    M=cv.moments(contours[i])
                    cX=int(M["m10"]/M["m00"])
                    cY=int(M["m01"]/M["m00"])
                    x[i]=cX
                    y[i]=cY
                    cv.circle(drawing,(cX,cY),7,color,-1)
                    cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
            # Show in a window
            return drawing
        time=0

        def get_area():
            #获取截取区域，若写入固定的范围则可以注释掉
            _,img=video.read()
            cv.namedWindow("click the boundary point")
            cv.setWindowProperty("click the boundary point",cv.WND_PROP_TOPMOST,1)
            cv.imshow("click the boundary point",img)
            xChoose=[]
            yChoose=[]
            def mouseEvent(event,x,y,flags,param):
                if event == cv.EVENT_LBUTTONDOWN:
                    xChoose.append(x)
                    yChoose.append(y)
                    xy="%d,%d"%(x,y)
                    cv.putText(img,xy,(x,y),cv.FONT_HERSHEY_PLAIN,1.0,(0,0,0),thickness=1)
                cv.imshow("click the boundary point",img)
            cv.setMouseCallback("click the boundary point",mouseEvent)
            cv.waitKey(0)
            cv.destroyAllWindows()
            xChoose.sort()
            yChoose.sort()
            print("[%d:%d,%d:%d]"%(yChoose[0],yChoose[-1],xChoose[0],xChoose[-1]))
            return xChoose,yChoose
        cv.namedWindow('preview')
        cv.setWindowProperty("preview",cv.WND_PROP_TOPMOST,1)
        cv.namedWindow('output')
        cv.setWindowProperty("output",cv.WND_PROP_TOPMOST,1)
        xChoose,yChoose=get_area()
        while True:
            time=time+1
            ret,oriframe=video.read()
            try:
                
                frame = oriframe[yChoose[0]:yChoose[-1],xChoose[0]:xChoose[-1]]
                if ret:
                    cv.imshow("previw",frame)
                    key=cv.waitKey(1)#如果不要看过程可以把这个注释掉，调整每帧之间的时间
                    if key == 83 or key == 115:
                        self.HSV,self.k_size,self.k_dilate_size,self.erode_times,self.dilate_times=filter(frame,self.HSV,
                        self.k_size[0],self.k_dilate_size[0],self.erode_times,self.dilate_times)
                    #     print(time/fps)
                    img2=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                    threshold=self.process(img2)
                    # img2=cv.blur(img2,k)
                    # img2=cv.GaussianBlur(img2,(5,5),-1)
                    #如果经常丢失目标erode可以减少一些，如果经常出现干扰点可以重新截取视频或者增加erode的次数
                    threshold=thresh_callback(threshold)#后面两个数代表开始/结束线性拟合的时间
                    cv.putText(threshold,str(round(time/fps*100)/100),(20,30),cv.FONT_HERSHEY_PLAIN,1.0,(255,255,255),thickness=1)
                    cv.imshow("output",threshold)
                else:
                    break
            except TypeError:
                print(traceback.format_exc())
                break
        # plt.plot(xLine,yLine)
        self.size=(yChoose[0],yChoose[-1],xChoose[0],xChoose[-1])
        cv.destroyAllWindows()
    def fine(self,starttime,speed):
        video=cv.VideoCapture(self.videoname)
        fps=video.get(cv.CAP_PROP_FPS)
        try:
            #这里很长的一段异常捕捉是为了在ctrl+c时能正常运行
            plt.figure()
            rng.seed(12345)
            fai_xLine=[]
            fai_yLine=[]
            last_delta_fai=0
            last_vector=np.array([0,0])
            filename=self.videoname
            print(filename)
            fps=video.get(cv.CAP_PROP_FPS)
            time=0

            cv.namedWindow('img')
            cv.setWindowProperty("img",cv.WND_PROP_TOPMOST,1)
            cv.namedWindow('output')
            cv.setWindowProperty("output",cv.WND_PROP_TOPMOST,1)
            while True:
                time=time+1
                ret,oriframe=video.read()
                try:
                    frame = oriframe[self.size[0]:self.size[1],self.size[2]:self.size[3]]
                    if ret:
                        cv.imshow("img",frame)
                        if time/fps>starttime:  #开始减速播放的时间
                            key=cv.waitKey(speed)
                            if key == 83 or key == 115:
                                self.HSV,self.k_size,self.k_dilate_size,self.erode_times,self.dilate_times=filter(frame,self.HSV,
                        self.k_size[0],self.k_dilate_size[0],self.erode_times,self.dilate_times)
                            if key == ord('q') or key == 27:
                                break
                        img2=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                        threshold=self.process(img2)
                        contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        # Draw contours
                        drawing = np.zeros((threshold.shape[0], threshold.shape[1], 3), dtype=np.uint8)
                        area=[]
                        if len(contours) >= 2:
                            for contour in contours:
                                area.append(cv.contourArea(contour)+rng.random())#加随机数是因为如果有俩面积大小一样，排序那步就会报错，也可以+index(contour)/100之类的？
                            # print(len(contours))
                            oricon=contours[:]
                            res= zip(area,oricon)
                            res=sorted(res,reverse=True)
                            _,contours=zip(*res)
                            x=[0,0]
                            y=[0,0]
                            for i in range(2):
                                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                                M=cv.moments(contours[i])
                                cX=int(M["m10"]/M["m00"])
                                cY=int(M["m01"]/M["m00"])
                                x[i]=cX
                                y[i]=cY
                                cv.circle(drawing,(cX,cY),7,color,-1)
                                cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
                            a=-1
                            new_vector=np.array([x[0]-x[1],y[0]-y[1]])
                            new_delta_fai=np.arccos(abs(new_vector.dot(last_vector)/(np.linalg.norm(new_vector)*np.linalg.norm(last_vector))))
                            if new_delta_fai<0.3 and abs(new_delta_fai-last_delta_fai)<0.05:
                                fai_xLine.append(time/fps)
                                fai_yLine.append(new_delta_fai)
                                last_delta_fai = new_delta_fai
                            last_vector=new_vector
                            
                        # Show in a window
                        cv.putText(threshold,str(round(time/fps*100)/100),(20,30),cv.FONT_HERSHEY_PLAIN,1.0,(255,255,255),thickness=1)
                        cv.imshow("output",drawing)
                    else:
                        break
                except TypeError:
                    print(traceback.format_exc())
                    break
            # plt.plot(xLine,yLine)
            plt.scatter(fai_xLine,fai_yLine,s=1)
            plt.title(filename)
            
        except KeyboardInterrupt:
            pass
        print("original boundary(ymin,ymax,xmin,xmax)")
        print(self.size)
        input_size=easygui.multenterbox("Enter in the new boundary","new boundary",["ymin","ymax","xmin","xmax"],self.size)
        output_size=[]
        for i in input_size:output_size.append(int(i))  #类型转换
        self.size=output_size
        plt.show()
        cv.destroyAllWindows()
        return
    def output(self,starttime,endtime,all_dot_print=False):
        video=cv.VideoCapture(self.videoname)
        fps=video.get(cv.CAP_PROP_FPS)
        plt.figure()
        queue_list=[]
        xfit_list=[]
        yfit_list=[]
        xLine=[]
        yLine=[]
        indexlist=[]
        for i in range(2,13):
            queue_list.append(queue.Queue(i))
            xfit_list.append([])
            yfit_list.append([])
            yLine.append([])
            xLine.append([])
            indexlist.append(i-2)
        rng.seed(12345)
        filename=self.videoname

        time=0
        def vector_angle(new_vector,last_vector):
            return np.arccos(abs(new_vector.dot(last_vector)/(np.linalg.norm(new_vector)*np.linalg.norm(last_vector))))
        size=self.size
        while True:
            time=time+1
            ret,frame=video.read()
            try:
                frame = frame[size[0]:size[1],size[2]:size[3]]
                if ret:
                    img2=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                    threshold = self.process(img2)
                    #如果经常丢失目标erode可以减少一些，如果经常出现干扰点可以重新截取视频或者增加erode的次数

                    #图像的轮廓寻找
                    contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    # Draw contours
                    drawing = np.zeros((threshold.shape[0], threshold.shape[1], 3), dtype=np.uint8)
                    area=[]
                    if len(contours) >= 2:
                        for contour in contours:
                            area.append(cv.contourArea(contour)+rng.random())#加随机数是因为如果有俩面积大小一样，排序那步就会报错，也可以+index(contour)/100之类的？
                        # print(len(contours))
                        oricon=contours[:]
                        res= zip(area,oricon)
                        res=sorted(res,reverse=True)
                        _,contours=zip(*res)
                        x=[0,0]
                        y=[0,0]
                        for i in range(2):
                            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                            M=cv.moments(contours[i])
                            cX=int(M["m10"]/M["m00"])
                            cY=int(M["m01"]/M["m00"])
                            x[i]=cX
                            y[i]=cY
                            cv.circle(drawing,(cX,cY),7,color,-1)
                            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
                        new_vector=np.array([x[0]-x[1],y[0]-y[1]])
                        for i in range(2,13):
                            if queue_list[i-2].full():
                                last_5_time,last_5_vector=queue_list[i-2].get()
                                new_delta_fai=vector_angle(last_5_vector,new_vector)
                                xLine[i-2].append(((last_5_time+time)/2/fps))
                                yLine[i-2].append(new_delta_fai*fps/(time-last_5_time))
                                if endtime > time/fps >starttime:
                                    xfit_list[i-2].append([((last_5_time+time)/2/fps)])
                                    yfit_list[i-2].append([new_delta_fai*fps/(time-last_5_time)])
                            queue_list[i-2].put((time,new_vector))
                else:
                    break
            except TypeError:
                print(traceback.format_exc())
                break
        model_list=[]
        modelR=[]
        for i in range(2,11):
                        model=LinearRegression()
                        model.fit(xfit_list[i],yfit_list[i])
                        if all_dot_print:plt.scatter(xLine[i],yLine[i],s=1)
                        modelR.append(model.score(xfit_list[i],yfit_list[i]))
                        model_list.append(model)
        res= zip(modelR,indexlist)
        res=sorted(res,reverse=True)
        modelR,indexlist=zip(*res)
        index=indexlist[0] 

        model=model_list[index]
        print(modelR[0])
        plt.plot(xfit_list[index],model.predict(xfit_list[index]),color="red")

        print(model.coef_[0][0])
        print(index)
        print(modelR)
        if not all_dot_print:plt.scatter(xLine[index],yLine[index],color="red",s=1)
        plt.title(filename)
        plt.ion()
        plt.show()
        img=cv.imread(consol_img_filepath)
        cv.namedWindow('consol')
        cv.setWindowProperty("consol",cv.WND_PROP_TOPMOST,1)
        cv.imshow("consol",img)
        while True:
            
            key=cv.waitKey(0)
            if key == 83 or key == 115:
                plt.cla()
                plt.title(self.videoname)
                plt.scatter(xLine[index],yLine[index],color="red",s=1)
            if key == 78 or key == 110:
                plt.plot(xfit_list[index],model.predict(xfit_list[index]),color="red")
            if key == 97 or key == 65:
                plt.cla()
                plt.title(self.videoname)
                for i in range(0,11):
                        plt.scatter(xLine[i],yLine[i],s=1)
            if 47<key<56:
                plt.scatter(xLine[key-48],yLine[key-48],s=1)
            if key == 56:
                plt.scatter(xLine[9],yLine[9],s=1)
            if key==57:
                plt.scatter(xLine[10],yLine[10],s=1)
            if key == 67 or key ==99:
                plt.cla()
                plt.title(self.videoname)
            if key == ord('q') or key == 27:
                plt.close()
                cv.destroyAllWindows()
                break
q=False
consol_img_filepath=easygui.fileopenbox("choose a picture you like as consol")
filepath=easygui.fileopenbox("choose the video")
print(filepath)
videoexp=experimentVideo(filepath)
videoexp.getRect()
while not q:
    choice=easygui.choicebox("What action?","experimentVideo",["getRect","fine","output"])
    #getRect:选取边界；fine:调整边界/图像处理参数；output:进行直线拟合并输出结果
    if choice=="getRect":
        videoexp.getRect()
    elif choice=="fine":
        starttime=easygui.integerbox("type in the start time of slowing",default=0)
        speed=easygui.integerbox("type in the interval between each flip",default=10)
        videoexp.fine(starttime,speed)
    elif choice=="output":
        while True:
            try:
                starttime=float(easygui.enterbox("type in the start time of output",default=0))
                break
            except:
                easygui.msgbox("please type in a float ")
        while True:
            try:
                endtime=float(easygui.enterbox("type in the end time of output",default=0))
                if endtime - starttime < 0.2:
                    easygui.msgbox("the end time must be after the starttime for a while")
                    continue
                break
            except:
                easygui.msgbox("please type in a float ")
        if starttime == 0:
            starttime=0.1
        print(starttime,endtime)
        videoexp.output(starttime,endtime)