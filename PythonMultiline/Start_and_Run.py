import threading
class myThread(threading.Thread):
    def __init__(self,threadID,name,counter):
        threading.Thread.__init__(self);
        self.threadID=threadID
        self.name=name
        self.counter=counter
    def run(self):
        currentThreadname=threading.current_thread()
        print("running in",currentThreadname)
thread=myThread(1,"mythrd",1)
#thread.run()
thread.start()
# 结果如下所示：
# running in <_MainThread(MainThread, started 1960)>
# running in <myThread(mythrd, started 7488)>
# 单纯调用thread.run()并不会去开启一个线程，还是属于简单的线程调用，在主线程中
# 而如果使用thread.start()，会开启一个线程，并且在开启的线程中执行对应的run方法
