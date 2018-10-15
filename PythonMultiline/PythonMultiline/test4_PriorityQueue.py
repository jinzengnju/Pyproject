#!/usr/bin/python
# -*- coding:UTF-8 -*-
# 线程优先级队列（ Queue）
import queue
import threading
import time
exitFlag=0

class MyThread(threading.Thread):
    def __init__(self,threadID,name,q):
        threading.Thread.__init__(self)
        self.threadID=threadID
        self.name=name
        self.q=q
    def run(self):
        print("Starting "+self.name)
        process_data(self.name,self.q)
        print("Exiting "+self.name)

def process_data(threadname,q):
    while not exitFlag:
        queueLock.acquire()
        #这里必须加上线程锁，这里锁住的不是资源，而是对代码片段提供线程锁
        #比如某时刻，队列还有一个元素，该元素正在被线程A取出，而与此同时线程B正在判断队列q是否为空，而此时线程B判断队列
        # 不为空而继续执行，但是当B取元素时，最后一个元素已经被A取出，造成线程等待，显示挂起
        #其他处理方式：通过加入q.get(timeout=10)超时操作来弥补这一问题。
        if not workqueue.empty():
            data=q.get()
            queueLock.release()
            print("%s processing %s"%(threadname,data))
        else:
            queueLock.release()
        time.sleep(1)

threadList=["Thread-1", "Thread-2", "Thread-3"]
nameList=["One", "Two", "Three", "Four", "Five"]
queueLock=threading.Lock()
workqueue=queue.Queue(10)
threads=[]
threadID=1

#线程创建
for tname in threadList:
    thread=MyThread(threadID,tname,workqueue)
    thread.start()
    threads.append(thread)
    threadID+=1
#队列填充
queueLock.acquire()
for word in nameList:
    workqueue.put(word)
queueLock.release()
# 等待队列清空
while not workqueue.empty():
    pass
#这个for循环与workqueue.join()效果相同

# 通知线程是时候退出
exitFlag = 1

# 等待所有线程完成
for t in threads:
    t.join()
print("Exiting Main Thread")












