import time
import threading
from queue import Queue
import numpy as np
import multiprocessing as mp


def worker(indx, task_queue, output_queue):
    done = 0
    while True:
        if task_queue.empty(): break
        _ = task_queue.get()
        if _==-1:
            print("thread",indx,' -1 signal recieved;', done,'tasks done')
            task_queue.task_done()
            break
        done+=1
        print("thread",indx,'works on task #',_)
        time.sleep(0.01)
        task_queue.task_done()
        print("thread",indx,'done task #',_)
        output_queue.put([indx, _, np.random.randint(1000)])
        
        # task_queue.task_done()

def start_thread(num_worker_threads, num_tasks):

    task_queue = mp.JoinableQueue()
    output_queue = mp.JoinableQueue()

    for i in range(num_tasks):
        task_queue.put(i)
        print('Queued task',i+1)
    for i in range(num_worker_threads):
        task_queue.put(-1)

    threads = []
    for index in range(num_worker_threads):
        thread = mp.Process(target=worker, args=(index, task_queue, output_queue))
        threads.append(thread)
    for p in threads:
        p.start()

    # for i, t in enumerate(threads):
    #     print('About to join thread',i,t)
    #     t.join()
    #     print('thread done jone',i,t)
    task_queue.join()
    print('reading outputs:')
    while not output_queue.empty():
        item = output_queue.get()
        print(item)
if __name__ == "__main__":
    start_thread(12, 100)