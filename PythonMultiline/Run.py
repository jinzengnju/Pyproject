import threading
import time
def worker():
    count = 1
    while True:
        if count >= 6:
            break
        time.sleep(1)
        count += 1
        print("thread name = {}, thread id = {}".format(threading.current_thread().name,threading.current_thread().ident))

t1=threading.Thread(target=worker,name="t1")
t2=threading.Thread(target=worker,name="t2")

t1.run()
t2.run()
# 两个线程都用run方法启动，但却是线运行t1.run，运行完之后才执行t2.run
# 两个线程都工作在主线程，没有启动新线程。因此，run方法仅仅是普通函数调用

print("----end----")