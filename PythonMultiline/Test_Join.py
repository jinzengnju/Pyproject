#!/usr/bin/python
# -*- coding:UTF-8 -*-
import threading
import time
class myThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
    def run(self):
        time.sleep(5)
        print(self.name)
def test1():
    print("main start")
    thread = myThread(1, "Thread-1")
    thread.start()
    # thread.join()
    print("main end")
    # main线程要等到t1线程运行结束后，才会输出“main end”。如果不加t1.join(),main线程和t1线程是并行的。
    # 而加上t1.join(),程序就变成是顺序执行了。

def test2():
    print("main start")
    t1=myThread(1,"thread-1")
    t2=myThread(2,"thread-2")
    t1.start()
    #等待t1结束，这时候t2线程并未启动
    t1.join()
    t2.start()
    t2.join()
    print("main end")
    #上面的两个子线程仍然是串行的，并没有达到真正意义上的并行

def test3():
    #这里可以让t1与t2并行
    print("main start")
    t1 = myThread(1, "thread-1")
    t2 = myThread(2, "thread-2")
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("main end")


if __name__=='__main__':
    test1()