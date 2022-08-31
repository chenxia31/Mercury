# ====================================================================#
# ==========================数组双指针====================================#
# 
# 【算法概述】
# 在遍历元素的过程中，不是使用单个指针进行访问，而是使用双指针来访问达到目的
# 
# 【算法思想】
# 根据指针的方向，可以分为
# 对撞指针：两个指针的方向相反
# 快慢指针：指针方向相同
# 分离双指针：如果两个指针分别属于不同的数组或者链表
# 
# =====================================================================#
# 
#

def threeSum(nums):
    res=[]
    def twoSum(arr,target):
        left=0
        print(arr)
        print(target)
        right=len(arr)-1

        while(left<right):
        	temp_sum=arr[left]+arr[right]
        	if temp_sum==target:
        		res.append([target*(-1),arr[left],arr[right]])
        		left+=1
        		right-=1
        	elif temp_sum<target:
        		left+=1
        	elif temp_sum>target:
        		right-=1
    nums.sort()

    for i in range(len(nums)):
        if nums[i]>0:
            break
        if i>0 and nums[i]==nums[i-1]:
        	continue
        twoSum(nums[i+1:],(-1)*nums[i])        
    return res
nums = [-1,0,1,2,-1,-4]
# print(threeSum(nums))
# 
# 
def threeSumClosest(nums, target):
    res=[]

    def twoSum(arr,tar):
        left=0
        right=len(arr)-1
        temp_odd=tar-arr[left]-arr[right]

        while(left<right):
            temp_sum=arr[left]+arr[right]
            if temp_sum==tar:
                temp_odd=0
                break
            elif temp_sum<tar:
                left+=1
            elif temp_sum>tar:
                right-=1
            if abs(temp_odd)>abs(tar-temp_sum):
                temp_odd=tar-temp_sum
        return temp_odd
        
    nums.sort()
    for i in range(len(nums)):
        if i<=len(nums)-3:
            odd=twoSum(nums[i+1:],target-nums[i])
            res.append(odd)
    # 寻找res的最小值
    min_odd=0
    minres=res[0]
    for i in range(len(res)):
    	if abs(res[i])<abs(minres):
    		minres=abs(res[i])
    		min_odd=i
    return target-res[min_odd]
nums=[1,1,1,0]
target=-100
# print(threeSumClosest(nums,target))
# 
# 

def findClosestElements(arr, k, x):
    # 首先二分查找，大于等于这个数组的第一个数下标
    def binarySearch(arr,target):
        left=0
        right=len(arr)-1
        res=arr[right]
        while(left<right):
            mid=left+(right-left)//2
            if arr[mid]==target:
                return mid,mid
            elif arr[mid]>=target:
                right=mid
            elif arr[mid]<target:
                left=mid+1
        return left-1,left

    close_left,close_right=binarySearch(arr,x)
    print(close_left)
    print(close_right)

    for i in range(k):
    	if close_left!=0 and close_right!=len(arr):
	        if abs(arr[close_left]-x)<=abs(arr[close_right]-x):
	            # 选择close_left加入
	            close_left-=1
	        else:
	        	close_right+=1
	    else:
	    	if close_left==0:
        		close_right+=1
        	else:
        		close_left-=1
    return arr[close_left:close_right+1]
arr=[-2,-1,1,2,3,4,5]
k=7
x=0
print(findClosestElements(arr,k,x))


