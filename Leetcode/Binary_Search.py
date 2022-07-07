# ====================================================================#
# ==========================二分查找====================================#
# 
# 【算法概述】
# 确定待查找元素所在的区间范围，再逐步缩小范围，直到找到元素或者找不到该元素为止
# 
# 【算法思想】
# 经典的减而治之的思想，减小问题规模来解决问题，每一次查找排除掉一定不存在目标元素的
# 区间，在剩下可能存在的目标元素的区间中继续查找，每一次通过一些条件判断，将待搜索的
# 区间逐渐缩小，来达到减少问题规模的目的
# 
# 【算法过程】
# 1. 每次查找从数组的中间元素开始，如果中间元素正好是查找的元素，则搜索过程结束
# 2. 如果特定元素大于或者小于中间元素，则在大于或者小于的元素中查找
# 3. 如果在某一步骤数组为空则代表找不到
# 
# 【算法重点】
# 1. 区间的开闭问题？
# 2. mid的取值问题
# 3. 出界条件的判断？
# 4. 搜索区间的范围选择问题？
# =====================================================================#
# 

def baseBinarySearch(nums,target):
	# nums是一个升序
	# 存在下标就返回
	# 不存在就返回-1
	# 因为要返回下标，所以采用index采用
	left=0
	right=len(nums)-1
	#=========================================#
	# 【二分查找的开闭问题】
	# 第一种：左闭右闭，区间中所有的点都可以得到
	# 第二种：左闭右开，右边界的点不能被取到
	#=========================================#

	while(left<=right):
	#=========================================#
	# 【出界条件的判断】
	# left<=right
	# 说明查找的元素不存在
	# left<right
	# 此时查找的区间不
	# 
	#=========================================#
		mid=(left+right)//2
	#=========================================#
	# 【MID的取值问题】
	# 第一种：(left+right)//2
	# 第二种：left+（right-left）//2
	# 前者是常见写法，后者是为了防止整型溢出。//2的代表的中间数是向下取整
	# 同时这种倾向于寻找左边的数组，这里可以选择
	# mid=(left+right+1)//2或者mid=left（right-left+1）//2
	# 同时这里的查找过成功1/2，也可以选择靠左一点、或者靠右一点
	# 从一般的意义上来说，趣中间位置元素在平均意义中达到的效果最好
	#=========================================#
		if nums[mid]<target:
	#=========================================#
	# 【搜索区间的选择】
	# 直接法，在循环体中元素之间返回
	# 排除法，在循环体中排出目标元素一定不存在区间
	#=========================================#

			# 说明target在右边
			left=mid+1
		elif nums[mid]>target:
			# 说明target在左边
			right=mid-1
		else:
			return mid

	return -1

nums=[1,1,2,3,4,44]
target=4
# print(baseBinarySearch(nums,target))


def searchRange(nums,target):
    res_left=-1
    res_right=-1

    left=0
    right=len(nums)-1

    # 找到第一个大于等于target的数字
    while(left<=right):
        mid=left+(right-left)//2
        if nums[mid]>=target:
            right=mid
            if left==right:
                res_left=left
                break
        elif nums[mid]<target:
            left=mid+1

    left=0
    right=len(nums)-1
    # 找到第二个大于等于target的数字
    while(left<=right):
        mid=left+(right-left+1)//2
        if nums[mid]>target:
            right=mid-1
        elif nums[mid]<=target:
            left=mid
            if left==right:
                res_right=right
                break
    return [res_left,res_right]
nums=[8,8,8,8,8,10]
target=8
# print('---')
# print(searchRange(nums,target))
# 


def findMin(nums):
    left=0
    right=len(nums)-1

    while(left<=right):
        # 找到第一个左边数比右边数小的数下标
        mid=left+(right-left)//2
        if mid==len(nums)-1:
            return nums[0]
        else:
            if nums[mid+1]>nums[mid]:
                # 需要更新
                if nums[mid]>=nums[0]:
                    left=mid+1
                else:
                    right=mid-1
            else:
                return nums[mid+1]
        print(left)
        print(right)
nums = [2,1]
# print(findMin(nums))

def findDuplicate(nums):
    # 注意分清楚 值 和 下标之间的区别
    left=1
    right=len(nums)-1
    temp_left=0
    temp_right=0
    while(left<=right):
        temp_left=0
        temp_right=0
        mid=(left+right)//2

        # 计算小于或者大于mid的数目
        for num in nums:
            if num<mid:
                temp_left+=1
            if num>mid:
                temp_right+=1
        # 接下来就是如何转移
        print(temp_right)
        if temp_left>mid-1:
            # 说明在左边
            right=mid-1
            
        elif temp_right>len(nums)-1-mid:
            # 说明在右边
            left=mid+1
            
        else:
            return mid
# print(findDuplicate([1,2,3,4,3]))


# 
print(2**(31)-1)