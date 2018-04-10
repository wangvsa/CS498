'''
Simple script to test how bitonic sort works
'''
import numpy as np

def bitonic_merge(arr):
    pass

def bitonic_sort(arr):
    n = 1       # number of ascending-descending array, denoted as ad_arr
    while n < len(arr):
        print(n)
        for i in range(n):
            ad_arr_len = len(arr) / n / 2
            start_i = i * ad_arr_len

        n = n * 2



if __name__ == "__main__":
    arr = np.arange(16)
    np.random.shuffle(arr)
    print("unsorted array:", arr)

    #bitonic_merge(arr)
    bitonic_sort(arr)

