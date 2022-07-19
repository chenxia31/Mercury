# class Queue:
# 	def _init_(self,size=100):
# 		print('hello world')
# 		self.size=size
# 		self.rear=-1
# 		self.front=-1
# 		self.queue=[None for _ in range(size)]

# 	def is_empty(self):
# 		# 这个时候需要区分队列非空和队列已满的情况
# 		return self.rear==self.front

# 	def is_full(self):
# 		return self.rear+1==self.size

# 	def enqueue(self,val):
# 		# 进队
# 		if self.is_full():
# 			return None
# 		else:
# 			self.rear+=1%self.size
# 			self.queue[rear]=val

# 	def dequeue(self,val):
# 		# 出队
# 		if self.is_empty():
# 			return None
# 		else:
# 			self.front+=1%self.size
# 			return self.queue[self.front]

# 	def get_rear(self):
# 		if self.is_empty():
# 			return None
# 		else:
# 			reutrn self.queue[self.rear]

# 	def get_front(self):
# 		if self.is_empty():
# 			return None
# 		else:
# 			return self.queue[(self.front+1)%self.size]


## 堆排序
# 最大堆排序
# def sortArray(nums):
# 	def heapfiy(arr,index,end):
# 		# 调整成为大顶堆
# 		left=index*2+1
# 		right=left+1
# 		while left<=end:
# 			max_index=index
# 			if arr[left]>arr[max_index]:
# 				max_index=left
# 			if right<=end and arr[right]>arr[max_index]:
# 				max_index=right
# 			if index== max_index:
# 				break 

# 			arr[index],arr[max_index]=arr[max_index],arr[index]
			
# 			# break
# 			index=max_index
# 			left=index*2+1
# 			right=left+1
# 		print(arr)

# 	def buildMaxHeap(arr):
# 		size=len(arr)
# 		# 堆里面最重要的一个性质就是，子节点的下标等于i*2+1、i*2+2
# 		# 最后一个非叶节点（size-2)//2
# 		for i in range((size-2)//2,-1,-1):
# 			heapfiy(arr,i,size-1)
# 		return arr

# 	def maxHeapSort(arr):
# 		# 原始序列构建大顶堆
# 		# 交换最大值和n-1的顺序
# 		# 新的序列重新构建大顶堆
# 		arr=buildMaxHeap(arr)
# 		size=len(arr)
# 		for i in range(size):
# 			arr[0],arr[size-i-1]=arr[size-i-1],arr[0]
# 			heapfiy(arr,0,size-i-2)
# 		return arr

# 	return maxHeapSort(nums)

def sortArray(nums):
	def heapfiy(arr,index,end):
		# 调整成为大顶堆
		left=index*2+1
		right=left+1
		while left<=end:
			min_index=index
			if arr[left]<arr[min_index]:
				min_index=left
			if right<=end and arr[right]<arr[min_index]:
				min_index=right
			if index == min_index:
				break 

			arr[index],arr[min_index]=arr[min_index],arr[index]
			

			# break
			index=min_index
			left=index*2+1
			right=left+1

	def buildMinHeap(arr):
		size=len(arr)
		# 堆里面最重要的一个性质就是，子节点的下标等于i*2+1、i*2+2
		# 最后一个非叶节点（size-2)//2
		for i in range((size-2)//2,-1,-1):
			heapfiy(arr,i,size-1)
		return arr

	def minHeapSort(arr):
		# 原始序列构建大顶堆
		# 交换最大值和n-1的顺序
		# 新的序列重新构建大顶堆
		arr=buildMinHeap(arr)
		print(arr)
		size=len(arr)
		for i in range(size):
			arr[0],arr[size-i-1]=arr[size-i-1],arr[0]
			heapfiy(arr,0,size-i-2)
		return arr

	return minHeapSort(nums)
# print(sortArray([23,34,23,1,3,9,667,9,0]))
# 
# 
# 

def topKFre(nums,k):
	num_dict=dict()
	for num in nums:
		if num in num_dict:
			num_dict[num]+=1
		else:
			num_dict[num]=1

	print(num_dict)
	new_nums=list(set(nums))

	def heapfiy(heap,index,end):
		child_l=index*2+1
		child_r=index*2+2
		while(child_l<=end):
			min_index=index
			if num_dict[heap[min_index]]>num_dict[heap[child_l]]:
				min_index=child_l
			if child_r<=end and num_dict[heap[min_index]]>num_dict[heap[child_r]]:
				min_index=child_r
			if num_dict[heap[index]]==num_dict[heap[min_index]]:
				break

			# 交换堆里面的位置
			heap[min_index],heap[index]=heap[index],heap[min_index]
			index=min_index
			child_l=index*2+1
			child_r=index*2+2

	def heappush(heap,val):
		heap.append(val)
		i=len(heap)-1
		while((i-1)//2>=0):
			cur_root=(i-1)//2
			if num_dict[heap[cur_root]]<num_dict[val]:
				break
			heap[i]=heap[cur_root]
			i=cur_root
		heap[i]=val
		return heap

	def heappop(heap):
		heap[0],heap[-1]=heap[-1],heap[0]
		heap.pop()
		heap=heapfiy(heap,0,len(heap)-1)
		return heap

	# 维护一个堆，每次push一个元素
	# 长度超过k，需要pop一个元素
	# 需要维护的是最小堆
	
	res=[]
	for num in new_nums:
		heappush(res,num)
	print(res)

	while(len(res)>k):
		heappop(res)
		print(res)
	return res
nums=[4,1,-1,2,-1,2,3]
k=2
print(topKFre(nums,k))



	








	