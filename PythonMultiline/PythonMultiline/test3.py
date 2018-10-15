#!/usr/bin/python
# -*- coding:UTF-8 -*-
#线程同步的问题，两个线程访问统一资源
#等待线程1执行结束后在执行线程2
import threading
import time
class MyThread(threading.Thread):
    def __init__(self,ThreadID,name,counter):
        threading.Thread.__init__(self)
        self.ThreadID=ThreadID
        self.name=name
        self.counter=counter
    def run(self):
        print("Staring "+self.name)
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        threadLock.acquire()
        print_time(self.name,self.counter,3)
        threadLock.release()

def print_time(threadName,delay,counter):
    while counter:
        time.sleep(delay)
        print("%s: %s"%(threadName, time.ctime(time.time())))
        counter-=1

threadLock=threading.Lock()
threads=[]
# 创建新线程
thread1 = MyThread(1, "Thread-1", 1)
thread2 = MyThread(2, "Thread-2", 2)

# 开启新线程
thread1.start()
thread2.start()

# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)
for t in threads:
    t.join()
print("Exsting main thread")