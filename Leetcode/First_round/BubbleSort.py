def bubbleSort(arr):
	# 冒泡排序的思想
	# 相邻元素之间的比较和变换，将值较小的元素逐步从后面移到前面，值较大的元素从前面移到后面
	# 
	# 冒泡排序的步骤
	# 逐步将i和i+1元素相比较，如果大小不合适则交换，这样重复一次可以保证下标为n的值最大
	# 之后对n-2元素重复操作，一直到排序结束
	# 
	for i in range(len(arr)):
		for j in range(len(arr)-i-1):
			if arr[j]>arr[j+1]:
				arr[j],arr[j+1]=arr[j+1],arr[j]

	return arr

def selectSort(arr):
	# 选择排序的思想
	# 每一次排序中，从剩余未排序元素中选择一个最小的元素，未排好序的元素最前面的那个元素交换位置
	# 
	# 选择排序算法步骤
	# 在算法中设置整型变量i，既可以作为排序树木的计算、同时i也作为执行第i次排序的时候，参加排序的后n-i+1元素的位置
	# 整型变量 min_i记录最小元素的下标
	# 结束之中交换两者之间的顺序
	for i in range(len(arr)-1):
		min_i=i
		for j in range(i+1,len(arr)):
			if arr[j]<arr[min_i]:
				min_i=j

		if i!=min_i:
			arr[i],arr[min_i]=arr[min_i],arr[i]

	return arr

def insertSort(arr):
	# 插入排序的基本思想
	# 每一次排序中，将剩余无序列序列的第一个元素，插入到有序序列的适当位置上
	# 
	# 插入排序的基本步骤
	# 将第一个元素看作一个有序序列
	# 从头到尾扫描无序序列，将扫描到的每个元素插入到有序序列的适当位置上
	for i in range(1,len(arr)):
		temp=arr[i]
		j=i
		# 0-（i-1）都是有序数组
		while j>0 and arr[j-1]>temp:
			arr[j]=arr[j-1]
			# 因为肯定需要移动一个位置
			j-=1
		arr[j]=temp
	
	return arr
				

def shellSort(arr):
	# 希尔排序的基本思想
	# 按照一定的间隔取值划分为若干个子序列，每个子序列按照插入排序，然后逐渐缩小间隔进行
	# 下一轮划分子序列和插入排序，一直到最后一轮排序间隔为1
	# 
	# 希尔排序是在插入排序的基础上进行改进的，因为我们可以看出插入排序在已经排好序的效率非常高
	# 但是插入排序效率比较低的原因是每次只能将数据移动以为
	# 
	# 希尔排序的算法步骤
	# 1. 确定元素间隔Gap，将序列按照1开始划分为若干个子序列，之间元素的间隔为一个gap
	# 2. 减少间隔数，并重新将整个序列按照新的间隔数分成若干个子序列，并对每个子序列进行排序
	# 
	# 
	size=len(arr)
	gap=size//2

	while gap>0:
		for i in range(gap,size):
			temp=arr[i]
			j=i
			while j>=gap and arr[j-gap]>temp:
				arr[j]=arr[j-gap]
				j-=gap
			arr[j]=temp
		gap=gap//2
	return arr


def mergeSort(arr):
	# 归并排序的基本思想：
	# 采用经典的分治策略，先递归将当前序列平均分成两半，然后将有序序列合并，最终合并成一个有序序列
	# 
	# 【算法步骤】
	# 1. 将数组中的所有数据堪称n有序的子序列
	# 2. 将当前序列组中的有序序列两两归并，完成一遍之后序列组里的排序序列的个数减版，每个子序列的长度加倍
	# 3. 重复上述操作得到一个长度为n的有序序列
	# 
	def merge(left_arr,right_arr):
		arr=[]
		while left_arr and right_arr:
			if left_arr[0]<=right_arr[0]:
				arr.append(left_arr.pop(0))
			else:
				arr.append(right_arr.pop[0])

		while left_arr:
			arr.append(left_arr.pop(0))

		while right_arr:
			arr.append(right_arr.pop(0))

		return arr

	size =len(arr)
	
	# 边界情况
	if size<2:
		return arr

	mid =size//2

	left_arr,right_arr=arr[0:mid],arr[mid:]
	return merge(mergeSort(left_arr),mergeSort(right_arr))

# def quickSort(arr,low,high):
# 	# 【基本思想】快速排序
# 	# 通过一趟排序将无序序列分为两个独立的序列，第一个序列的值均比第二个序列小，然后递归排列两个子序列
# 	# 
# 	# 【算法步骤】
# 	# 从数组中找到一个基准数
# 	# 将数组中比数大的元素移动到基准数右侧，比小的元素移动到基准数左侧，将数组拆分为左右两个部分
# 	# 重复两步骤
# 	# 
# 	def sort_index(arr,low,high):
# 		i=low-1
# 		pivot=arr[high]
      
# 	def random_index(arr,low,high):
# 		i=(low+high)//2
# 		arr[i],arr[high]=arr[high],arr[i]
# 		return sort_index(arr,low,high)

# 	if low<high:
# 		pi=random_index(arr,low,high)
# 		quickSort(arr,low,pi-1)
# 		quickSort(arr,pi+1,high)

# 	return	arr

def quickSort(arr,low,high):
	# 快速排序，排序的边界条件 low比high小
	
	def resort(arr,low,high):
		# 这里的数组中，目标数已经放到的最右边
		# 现在需要把大于目标值的数据放到右边
		
		flag=low-1

		for j in range(low,high+1):
			# 双指针
			if arr[j]<=arr[high]:
				flag+=1
				arr[flag],arr[j]=arr[j],arr[flag]

		return flag


	def random_index(arr,low,high):
		# 当然还需要进行一些小的操作
		index=(low+high)//2

		# 默认左边都是最小的
		arr[index],arr[high]=arr[high],arr[index]
		return resort(arr,low,high)

	if low<high:
		# 每次按照队列排序
		# 需要返回标杆的下标
		index =random_index(arr,low,high)

		# 之后还需要继续分解
		quickSort(arr,low,index-1)
		quickSort(arr,index+1,high)
	return arr

def heapSort(arr):
	# 借用堆结构所设计的排序算法，将数组转换为大顶堆，重复从大顶堆中取出数值最大的节点，并让剩余的堆维持大顶堆的性质
	# 
	# 【堆的定义】
	# 大顶堆，根节点值大于子节点值
	# 小顶堆，根节点值小于等于子节点值
	# 
	# 【算法步骤】
	# 1. 首先将无序序列构造成第1个大顶堆，使得m个元素的最大值在序列的第一个值
	# 2. 交换序列的最大值元素与最后一个元素的位置
	# 3. 将前面n-1元素组成的序列调整称为一个新的大顶堆，这样得到第2个最大值元素
	# 4. 如此循环下去，知道称为一个有序序列
	# 
	arrLen =len(arr)

	def heapify(arr,i):
		left=2*i+1
		right=2*i+2
		largest=i
		if left<arrLen and arr[left]>arr[largest]:
			largest=left
		if right<arrLen and arr[right]>arr[largest]:
			largest=right

		if largest!=i:
			arr[i],arr[largest]=arr[largest],arr[i]
			heapify(arr,largest)
		print(arr)

	def buileMaxHeap(arr):
		for i in range(len(arr)//2,-1,-1):
			heapify(arr,i)

	buileMaxHeap(arr)
	print('buil')
	for i in range(arrLen-1,0,-1):
		arr[0],arr[i]=arr[i],arr[0]
		arrLen-=1
		heapify(arr,0)

	return arr

def countingSort(arr):
	# 【基本思想】
	# 使用一个额外的数组counts，其中counts元素是排序数组中arr等于i的个数
	# 根据数组counts来将arr的元素排列到正确位置
	min_arr,max_arr=min(arr),max(arr)

	counts =[0 for _ in range(max_arr-min_arr+1)]

	for num in arr:
		counts[num-min_arr]+=1

	for j in range(1,max_arr-min_arr+1):
		counts[j]+=counts[j-1]

	res=[0 for _ in range(len(arr))]
	for i in range(len(arr)-1,-1,-1):
		res[counts[arr[i]-min_arr]-1]=arr[i]
		counts[arr[i]-min_arr]-=1
	return res

def raidxSort(arr):
    # 基数排序radix sort【基本思想】
    # 将整数按照位切割称为不同的数字，然后按照个位数来从小到达进行排列
    # 注意这个排序方式是先比较个位数，然后逐渐向高位数
    
    # 首先需要了解到基数排序中最大位数
    max_radix=0
    for num in arr:
    	if num>0:
    		temp=len(str(num))
    		if max_radix<temp:
    			max_radix=temp
    	else:
    		temp=len(str(num))-1
    		if max_radix<temp:
    			max_radix=temp

    for radix in range(max_radix):
        # 从最低位到最高位开始计算
        # 按照位数来生成
        buckets=[[] for _ in range(10)]
        # 按照个位数，放到每个篮子，再按照十位数放到篮子
        for num in arr:
            # 提取radix对应的位数，0代表个位
            # 这里使用转换为字符串的方式来方便理解
            
            if len(str(abs(num)))<(radix+1):
                # 位数不够的时候，说明这个位置没有0，就需要放到第一格
                buckets[0].append(num)
            else:
                index=str(abs(num))[len(str(abs(num)))-1-radix]
                buckets[int(index)].append(num)
        # 提取到buckets之后需要重新解析arr中
        arr.clear()
        for bukcet in buckets:
            for num in bukcet:
                arr.append(num)
    neg_arr=[]
    pos_arr=[]	

    for num in arr:
    	if num>0:
    		pos_arr.append(num)
    	else:
    		neg_arr.append(num)
    return neg_arr[::-1]+pos_arr


def newquickSort(arr,low,high):
    '''
    代码补充
    '''
    def reSort(arr,low,high):
        res=low-1

        for j in range(low,high+1):
            if arr[j]<=arr[high]:
                res+1
                arr[res],arr[j]=arr[j],arr[res]
        return res
    
    def random_index(arr,low,high):
        mid=low+(high-low)//2

        arr[high],arr[mid]=arr[mid],arr[high]
        return reSort(arr,low,high)
    
    if low<high:
        index=random_index(arr,low,high)
        quickSort(arr,low,index-1)
        quickSort(arr,index+1,high)
        print(index)
    return arr

test_arr=[1280, -3627, -2475, -8776, -3166, 6680]
import numpy as np
test_arr=np.random.permutation(20)
# print(bubbleSort(test_arr))
# print(selectSort(test_arr))
# print(insertSort(test_arr))
# print(shellSort(test_arr))
# print(mergeSort(test_arr))
print(newquickSort(test_arr,0,len(test_arr)-1))
# print(heapSort([1,2,4,3,5]))
# print(countingSort(test_arr))



