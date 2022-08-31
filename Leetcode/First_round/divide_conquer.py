# https://blog.csdn.net/zhongkelee/article/details/44901905
# 分置法的基本思想上将问题划分为多个子问题，这样递归下去直到问题足够小
# 将求出来的小规模问题的解合并成一个更大规模的问题的解，自底向上逐步求出原问题的解
# 
# 递归的【基本思想】
# 直接或者间接调用自身的算法称为递归算法
# 用函数自身给出定义的函数称为递归函数
# 递归是算法的实现方式、分治是算法的设计思想
# 迭代是不断循环过程、递归是不断的调用自身
# 
# 
import sys

sys.setrecursionlimit(10000) # set the maximum depth as 40000

def step_multi(n):
	# 边界条件和递归函数是两要素
	# 只有具备这两个要素，才能在有限次计算后得到结果
	if n==1:
		return n
	else:
		return n*step_multi(n-1)

def fibonacci_set(num):
	# 边界条件
	if num==0:
		return 1
	elif num==1:
		return 1
	else:
		return fibonacci_set(num-1)+fibonacci_set(num-2)

def ackerman(nnum,mmnum):
	# 当一个函数及它的变量是由函数自身定义的这个函数是一个双递归函数
	if nnum==1 and mmnum==0:
		return 2
	if nnum==0 and mmnum>=0:
		return 1
	if nnum>=2 and mmnum==0:
		return nnum+2
	if nnum>=1 and mmnum>=1:
		return ackerman(ackerman(nnum-1,mmnum),mmnum-1)

def int_split(num,max_sub):
	# 将正整数表示成一系列正整数之后
	# 单纯的对n进行设置，很难找到递归关系，因此需要考虑增加一个自变量
	# 将最大加数n1不大于的m的划分个数计做int_split(n,m)
	# 我们需要建立的递归关系
	if max_sub==1:
		return 1
	if num<max_sub:
		return int_split(num,num)
	if num==max_sub:
		return 1+int_split(num,max_sub-1)
	if num>max_sub>1:
		return int_split(num-max_sub,max_sub)+int_split(num,max_sub-1)

def mergeSort(arr):
	def merge(left_arr,right_arr):
		arr=[]

		# 两者都有值的情况
		while left_arr and right_arr:
			# 哪个大就放那个
			if left_arr[0]<right_arr[0]:
				arr.append(left_arr[0])
				left_arr.pop(0)
			else:
				arr.append(right_arr[0])
				right_arr.pop(0)

		# 可能存在某一个没有了
		while left_arr:
			arr.append(left_arr[0])
			left_arr.pop(0)

		while right_arr:
			arr.append(right_arr[0])
			right_arr.pop(0)
		return arr

	if len(arr)==1:
		return arr
	else:
		mid=len(arr)//2
		left_arr,right_arr=arr[:mid],arr[mid:]
		return merge(mergeSort(left_arr),mergeSort(right_arr))

test_arr=[534,6564,23,534,2,6,342,653,88,98,354,667,45,67,876,10,76,465,56,7,8,9,6,45,433,66]
print(mergeSort(test_arr))
print(int_split(6,6))