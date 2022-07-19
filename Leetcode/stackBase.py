# ====================================================================#
# ========================== 堆栈基础知识 ====================================#
# 
# 【基础知识】stack
# 简称为栈，是一种线性表数据结构，是一种只允许在表的一端进行插入和删除操作的线性表
# - 栈中允许插入和删除的一端称为栈顶 top；另一端称为bottom，当表中没有任何数据称为空栈
# 
# - 基本操作，插入操作（入栈/进栈）、删除操作（出栈/退栈）
# 
# - 后进先出的线性表（last in first out）
# 
# 【存储方式】
# 和线性表相似，栈有两种存储表示方式，顺序栈和链式栈
# - 顺序栈也就是利用list来实现
# - 链式栈就是使用linkedList来实现
# 
# 【基本操作】
# 插入和删除操作被改成入栈（push）和出栈（pop），其中的一些基本操作
# 1. 初始化空栈，定义你size和top指针
# 2. 判断栈是否为空
# 3. 判断栈是否已满
# 4. push
# 5. pop
# 6. 获取栈顶元素
# 
# 【基本使用】
# 1. 使用推展可以方便的保存和取用信息，因此通常被用作算法和程序中辅助存储结构，作为临时
# 保存信息，来方便后面操作的使用
# 2. 堆栈的后进先出元素可以保证特定的存取顺序，比如铁路列车车辆调度
# =====================================================================#
# 
# T227表达式求值
# 
# 在计算表达式中，乘除运算优先加减元算，可以进行乘除运算之后将整数放到表达式中相应位置来
# 计算加减
# 
# 遍历字符串s，使用变量op来标记数字之前的运算符，默认为‘+’
# 遇到数字向后遍历，得到num来判断op的符号
# 	+ 直接压栈
# 	- 取负后压栈
# 	* 取出栈顶元素后压栈
# 	/ 取栈顶元素在计算int（top/num)
def calculate(s):
	size=len(s)
	stack=[]
	op='+'
	index=0
	num=''
	while(index<size):

		if s[index] in '+-*/':
			# 如果是运算符
			num=''
			op=s[index]
			index+=1
		else:
			# 需要将num取出来
			while(index<size and s[index] not in '+-*/'):
				num+=s[index]
				index+=1
			num=int(num)
			if op=='+':
				stack.append(num)
			elif op=='-':
				stack.append(-num)
			elif op=='*':
				top=stack.pop()
				stack.append(top*num)
			elif op=='/':
				top=stack.pop()
				stack.append(int(top/num))
			
	return sum(stack)
# print(calculate('2+3/2+4'))

class MinStack:

    def __init__(self):
        self.stack=[]
        self.topPointer=-1
        self.min=[]


    def push(self, val: int) -> None:
        self.stack.append(val)
        self.topPointer+=1
        if self.topPointer==0:
            self.min.append(val)
        else:
            if val<self.min[-1]:
                self.min.append(val)
            else:
                self.min.append(self.min[-1])

    def pop(self) -> None:
        self.min.pop()
        self.stack.pop()
        self.topPointer-=1
        


    def top(self) -> int:

        return self.stack[-1]


    def getMin(self) -> int:
        return self.min[-1]
ms=MinStack()
ms.push(1)
ms.push(2)
# print(ms.top())
# print(ms.getMin())
# print(ms.stack)

# ====================================================================#
# ========================== 单调栈知识 ====================================#
# 
# 【基础知识】Monotone Stack
# 一种特殊的的栈，在栈的先进后出的基础上，要求栈顶到栈底的元素是单调递增（或者单调递减）
# 
#
# 
# 【单调递增栈】
# 只有比栈顶元素小的元素才能直接进栈，否则需要将栈内比当前元素小的元素出栈，再入栈；
# 这里就保证栈中保留的都是比当前入栈元素大的值
# 
# 单调栈可以在时间复杂度为O（n）求解出某个元素左边或者右边第一个比它大或者小的元素
# 
# 【背诵】
# 1. 无论哪种题型，都建议从左到右遍历元素
# 2. 查找比当前元素大的元素就用单调递增栈，查找比当前元素小的元素使用单调递减栈
# 3. 从左侧查找就看插入栈时候的栈顶元素，从右侧查找就看弹出栈即将插入的元素
# =====================================================================#

def monotoneIncreasingStack(nums):
	stack=[]
	for num in nums:
		while stack and num>=stack[-1]:
			stack.pop()
		stack.append(num)

# T496 下一个更大元素
def nextGreaterElement(nums1,nums2):
	# 第二种使用单调递增栈，因为nums1是num2的子集，所以可以遍历nums2
	# 构造单调递增栈，求出nums2每个元素右侧下一个最大的元素，然后存储在哈希表中
	# 
	# 【具体做法】
	# res存储答案，使用stack表示单调递增栈，使用哈希表num-map存储nums2中比下一个当前
	# 元素大的数值，当前数值：下一个比当前元素大的数值
	# 
	# 遍历nums2，对于当前元素，如果小则入栈，如果元素大则一直出栈，出栈元素是第一个大
	# 于当前元素值的元素
	# 
	# 遍历玩数组nums2周，建立好哈希表之后，遍历数组1
	# 
	# 从num-map中取出对应的值
	res=[]
	stack=[]
	num_map=dict()

	for num in nums2:
		while(stack) and num>stack[-1]:
			num_map[stack[-1]]=num
			stack.pop()
		stack.append(num)
		print(stack)

	for num in nums1:
		res.append(num_map.get(num,-1))
	return res
nums1=[4,1,2]
nums2=[1,7,9,4,5]
# print(nextGreaterElement(nums1,nums2))
# 

def removeDuplicateLetters(s):
	stack=[]
	letter_counts=dict()

	for char in s:
		if char in letter_counts:
			letter_counts[char]+=1
		else:
			letter_counts[char]=1

	for char in s:
		if char not in stack:
			while stack and ch<stack[-1] and stack[-1] in letter_counts:
				stack.pop()
			stack.append(ch)
		letter_counts[char]-=1

	return ''.join(stack)








