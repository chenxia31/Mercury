# =======================================================================#
# =========================数组滑动窗口====================================#
# 
# 【算法概述】
# 在计算机网络中，滑动窗口是传输层进行一种流量控制的措施，也就是接收方通过告诉发送方自己的
# 窗口大小来控制发送方的发送速度，这里也是利用相同的特性：
# 在给定数组或者字符串中维护一个固定长度或者不定长度的窗口，可以对窗口进行滑动操作、缩放操作
# 以及维护最优解操作
# 
# 滑动操作：窗口按照一定方向进行移动，最常见的是向右侧移动
# 缩放操作；对于不定长度的窗口，可以左侧缩小窗口长度，右侧增加窗口长度
# 
# 也是利用双指针中快慢指针技巧，
# 
# 【算法思想】
# 滑动窗口算法一般用来解决一些查找满足一定条件的连续区间长度的性质，这可以将问题中的嵌套循环
# 转变成为一个单循环，因此可以减少时间复杂度
# 
# 【第一种】：固定长度窗口
# 【第二种】：不定长度窗口
# 			求解最大的满足条件的窗口
# 			求解最小的满足条件的窗口
# 
# =====================================================================#
# 
# 
# T209 长度最小的子数组
def minSubArrayLen(target,nums):
	'''
	找到该数组中满足大于等于targeted的连续子数组，并返回其长度，如果不存在符合条件的子数组则返回0
	'''
	size=len(nums)
	ans=size+1
	left=0
	right=0
	window_sum=0

	while right<size:
		window_sum+=nums[right]

		while window_sum>=target:
			ans=min(ans,right-left+1)
			window_sum-=nums[left]
			left+=1

		right+=1

	return ans if ans!=size+1 else 0

target=0
nums=[1,2,3]
# print(minSubArrayLen(target,nums))

def numSubarrayProductLessTanK(k,nums):
	if k<=1:
		return 0

	size=len(nums)
	left=0
	right=0
	windows_product=1

	cont=0

	while right<size:
		windows_product*=nums[right]
		while windows_product>=k:
			windows_product/=nums[left]
			left+=1
		cont+=(right-left+1)
		right+=1
	return cont






