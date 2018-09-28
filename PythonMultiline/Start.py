import threading
import time
def worker():
    count = 1
    while True:
        if count >= 6:
            break
        time.sleep(1)
        count += 1
        print("thread name = {}, thread id = {}".format(threading.current_thread().name,
                                                        threading.current_thread().ident))
t1=threading.Thread(target=worker,name="t1")
#注意：threading.Thread方法只是创建了该线程，但实际上该线程还未运行
t2=threading.Thread(target=worker,name='t2')
t1.start()
t2.start()
#启动开启一个线程，线程名字为t1，target=worker表示线程函数，即线程需要执行的函数
#start()启动后会自动去执行传入的target线程函数以及run函数
print("------end---------")
#start()方法启动了两个新的子线程并交替运行，每个子进程ID也不同。