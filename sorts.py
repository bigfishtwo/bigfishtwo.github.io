import time
import random

class QuickSort:
    def __init__(self):
        pass
    def sort(self, nums):
        self.quickSort(nums, 0, len(nums) - 1)
        return nums

    def quickSort(self, nums, left, right):
        if left < right:
            pivot = self.partition(nums, left, right)
            self.quickSort(nums, left, pivot - 1)
            self.quickSort(nums, pivot + 1, right)

    def partition(self, nums, left, right):
        pivot = nums[right]
        i = left
        for j in range(left, right):
            if nums[j] < pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[right], nums[i] = nums[i], nums[right]
        return i


class MergeSort:
    def __init__(self):
        pass
    def mergesort(self, nums):
        if len(nums)<=1:
            return nums
        mid = len(nums)//2
        left = self.mergesort(nums[:mid])
        right = self.mergesort(nums[mid:])
        sorted = self.merge(left, right)
        return sorted
    def merge(self, left, right):
        l, r = 0,0
        merged = []
        while l<len(left) and r<len(right):
            if left[l]<right[r]:
                merged.append(left[l])
                l+=1
            else:
                merged.append(right[r])
                r +=1

        merged.extend(left[l:])
        merged.extend(right[r:])
        return merged

def bubbleSort(nums):
    # 时间复杂度: O(n^2)
    # 空间复杂度: O(1)
    # 稳定
    n = len(nums)
    for i in range(n-1):
        for j in range(n-i-1):
            if nums[j]>nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums

def selectSort(nums):
    # 时间复杂度: O(n^2)
    # 空间复杂度: O(1)
    # 不稳定
    n = len(nums)
    for i in range(n-1):
        minid = i
        for j in range(i+1,n):
            if nums[j]<nums[minid]:
                minid = j

        nums[i], nums[minid] = nums[minid], nums[i]
    return nums

def insertSort(nums):
    # 时间复杂度: O(n^2)
    # 空间复杂度: O(1)
    # 稳定
    n = len(nums)
    for i in range(1,n):
        prev = i - 1
        curr = nums[i]
        while prev>=0 and nums[prev]>curr:
            nums[prev+1] = nums[prev]
            prev -= 1
        nums[prev+1] = curr
    return nums

def shellSort(nums):
    n = len(nums)
    gap = n//2
    while gap:
        for i in range(gap,n):
            j = i
            curr = nums[i]
            while j-gap >=0 and curr<nums[j-gap]:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = curr
        gap = gap//2
    return  nums

def countSort(nums, maxv):
    sortedIndx = 0
    n = len(nums)
    bucklen = maxv + 1
    bucket = [0]*bucklen

    for i in range(n):
        bucket[nums[i]] +=1

    for i in range(bucklen):
        while bucket[i]:
            nums[sortedIndx] = i
            bucket[i] -= 1
            sortedIndx += 1
    return nums

def backpack(capity, values, weights):
    n = len(values)
    # dp = [[0]*(capity+1) for _ in range(n+1)]
    # for i in range(1,n+1):
    #     for c in range(1,capity+1):
    #         if weights[i-1]<=c:
    #             dp[i][c] = max(dp[i-1][c], dp[i-1][c-weights[i-1]]+values[i-1])
    #         else:
    #             dp[i][c] = dp[i-1][c]
    # return dp[-1][-1]
    # 空间优化，一维数组
    dp = [0]*(capity+1)
    for i in range(n):
        for j in range(capity,1,-1): # 注意倒着计算容量
            if j>=weights[i]:
                dp[j] = max(dp[j], dp[j-weights[i]]+values[i])
    return dp[-1]


def backpack_complete(capity, values, weights):
    n = len(values)
    # dp = [[0]*(capity+1) for _ in range(n+1)]
    # for i in range(1, n+1):
    #     for j in range(1,capity+1):
    #         if j >= weights[i-1]:
    #             dp[i][j] = max(dp[i-1][j], dp[i][j - weights[i-1]] + values[i-1])
    #         else:
    #             dp[i][j] = dp[i-1][j]
    # return dp[-1][-1]
    dp = [0] * (capity + 1)
    for i in range(n):
        for j in range(weights[i],capity+1):  # 注意倒着计算容量
            if j >= weights[i]:
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[-1]


if __name__ == '__main__':
    w = [3,2,5,1,6,4]
    v = [6,5,10,2,16,8]
    c = 10
    print(backpack_complete(c,v,w))