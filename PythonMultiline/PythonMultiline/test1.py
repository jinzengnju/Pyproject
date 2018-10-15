#!/usr/bin/python
# -*- coding:UTF-8 -*-
#通过函数式调用thread模块中的start_new_thread()函数创建新的线程
import time
import _thread
def print_time(threadName,delay):
    count=0
    while count<5:
        time.sleep(delay)
        count+=1
        print("%s:%s"%(threadName,time.ctime(time.time())))

try:
    _thread.start_new_thread(print_time,("Thread-1",2,))
    _thread.start_new_thread(print_time,("Thread-2",4,))
except:
    print("cannot start thread")
while 1:
    pass
#这里必须得有while(1)才能看到效果
#线程结束一啊不能依靠线程函数的自然结束，也可以在线程函数中调用thread.exit()