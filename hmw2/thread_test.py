import time
import threading
from queue import Queue
import numpy as np


def worker(indx, task_queue, output_queue):
    while True:
        if task_queue.empty(): break
        _ = task_queue.get()
        print("thread",indx,'works on task #',_)
        time.sleep(1)
        print("thread",indx,'done task #',_)
        output_queue.put([indx, _, np.random.randint(1000)])
        
        task_queue.task_done()

def start_thread(num_worker_threads, num_tasks):

    task_queue = Queue()
    output_queue = Queue()

    for i in range(num_tasks):
        task_queue.put(i)
        print('Queued task',i+1)

    threads = []
    for index in range(num_worker_threads):
        thread = threading.Thread(target=worker, args=(index, task_queue, output_queue))
        threads.append(thread)
        thread.start()

    for i, t in enumerate(threads):
        print('About to join thread',i,t)
        t.join()
        print('thread done jone',i,t)
    while not output_queue.empty():
        item = output_queue.get()
        print(item)
if __name__ == "__main__":
    start_thread(3, 10)