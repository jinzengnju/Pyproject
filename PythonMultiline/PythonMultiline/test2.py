#!/usr/bin/python
# -*- coding:UTF-8 -*-
# 使用Threading模块创建线程
# 使用Threading模块创建线程，直接从threading.Thread继承，然后重写__init__方法和run方法
import threading
import time
exitFlag=0
class MyThread(threading.Thread):
    def __init__(self,threadID,name,counter):
        threading.Thread.__init__(self)
        self.threadID=threadID
        self.name=name
        self.counter=counter
    def run(self):
        print("Starting"+self.name)
        print_time(self.name,self.counter,5)
        print("Exiting"+self.name)

def print_time(threadName,delay,counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print("%s: %s"%(threadName, time.ctime(time.time())))
        counter-=1


# 创建新线程
thread1 = MyThread(1, "Thread-1", 1)
thread2 = MyThread(2, "Thread-2", 2)

# 开启线程
thread1.start()
thread2.start()

print("Exiting main thread")