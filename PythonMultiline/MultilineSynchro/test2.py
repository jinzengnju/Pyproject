#!/usr/bin/python
# -*- coding:UTF-8 -*-
# 使用条件变量保持线程同步：
#一下两个线程主要是靠con条件变量来实现同步，通过线程wait始放锁进入等待池，然后notify唤醒相关线程获取锁后继续执行
import threading
class Producer(threading.Thread):
    def __init__(self,threadName):
        threading.Thread.__init__(self,name=threadName)
    def run(self):
        global x
        con.acquire()
        if x==1000:
            con.wait()
            pass
        else:
            for i in range(1000):
                x=x+1
                con.notify()
        print(x)
        con.release()

class Cosumer(threading.Thread):
    def __init__(self,threadname):
        threading.Thread.__init__(self,name=threadname)
    def run(self):
        global x
        con.acquire()#获取隐含锁
        if x==0:
            con.wait()#wait方法必须在线程获得锁的条件下执行，并且会释放隐形锁.
            #会阻塞到被其他线程调用此条件变量（这里是指con变量）的notify唤醒。一旦被唤醒，线程立即重新获取锁并返回
            #所以这里是靠Producer的notify来唤醒
            pass
        else:
            for i in range(1000):
                x=x-1
            con.notify()#通知其他线程
        print(x)
        con.release()

if __name__=='__main__':
    con=threading.Condition()
    x=0
    p=Producer('Producer')
    c=Cosumer('Cosumer')
    p.start()
    c.start()
    p.join()
    c.join()
    print(x)
