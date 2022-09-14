#Readme#######
# class Timer:benchmark for time,22-09-10
#
#

#######
# timer banchmark
# saver
import numpy as np
import time
class Timer:
    '''记录多次运行的时间'''
    def __init__(self) -> None:
        self.times=[]
        self.start()
    
    def start(self):
        '''启动定时器'''
        self.tik=time.time()
    def stop(self):
        '''停止计时并记录在列表中'''
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    def avg(self):
        '''return average time'''
        return sum(self.times)/len(self.times)
    def sum(self):
        '''return sum of the time'''
        return sum(self.times)
    def cumsum(self):
        '''返回累积时间'''
        return np.array(self.times).cumsum().tolist()
