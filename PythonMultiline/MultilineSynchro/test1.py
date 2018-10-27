#!/usr/bin/python
# -*- coding:UTF-8 -*-
# Thread的Lock和RLock实现简单的线程同步：
import threading
import time
class mythread(threading.Thread):
    def __init__(self,threadname):
        threading.Thread.__init__(self,name=threadname)
    def run(self):
        global x
        lock.acquire()
        for i in range(3):
            x=x+1
        time.sleep(1)
        print(x)
        lock.release()
#10个线程共同修改全局变量x，每个线程运行完了才会始放锁给另一个线程
if __name__=="__main__":
    lock=threading.Lock()
    t1=[]
    for i in range(10):
        t=mythread(str(i))
        t1.append(t)
    x=0
    for i in t1:
        i.start()