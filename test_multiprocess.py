
import multiprocessing
from tqdm import tqdm
class ExampleClass(object):
    def __init__(self, num1=10, num2=20):
        self.num1 = num1
        self.num2 = num2
        
    def add_num(self, num1=10, num2=10):
        self.check_num(num1)
        self.check_num(num2)
        return num1 + num2
    
    def check_num(self, num):
        if isinstance(num, int):
            print("{num} is an int")
        else:
            print("{num} is not an int")
        
    
    
class MultiprocClass(ExampleClass):
    def __init__(self):
        super().__init__()
        
def multiproc_wrapper(args):
    multi_proc_classinit = MultiprocClass()
    res = multi_proc_classinit.add_num(**args)
    return res


args = [{"num1": num, "num2": num*2} for num in range(10, 100)]
num_processes = multiprocessing.cpu_count()
chunksize = max(1, len(args) // num_processes)
with multiprocessing.Pool(num_processes) as p:
        results = list(
                    tqdm(
                        p.imap_unordered(
                            multiproc_wrapper, args, chunksize=chunksize
                        ),
                        total=len(args),
                    )
                )
        
if __name__ == "__main__":
    print(results)