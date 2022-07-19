# ====================================================================#
# ======================链表基础知识====================================#
# 
# 【链表简介】linked list
# 一种线性表数据结构，使用一组任意的存储单元来存储一组具有相同类型的数据
# 
# 链表的优点是存储空间不必事先分配，在需要存储空间的时候可以临时申请，不会造成空间的浪费；
# 在插入、移动和删除元素的时间效率中比数组高
# 
# 缺点：不仅数据元素本身的数据信息要占用存储空间，指针也需要占用存储空间，链表结构比数组
# 结构的空间开销大
# 
# 【链表分类】
# 1. 单链表
# 
# 后继指针next链接下一个节点
# 
# 2. 双链表 doubly linked list 
# 
# 每个链节点中有两个指针，分别指向直接后续和直接前驱
# 从双链表中的任意一个节点开始，都可以很方便的访问前驱节点和后继节点
# 
# 3. 循环链表 circular linked list
# 链表的一种，将最后一个链节点指向头节点，形成一个环
# =====================================================================#

# 2.1 链表的结构定义
# 
# 首先定义一个链表的节点，使用val变量表示数据元素的值、使用指针变量next表示后继指针

class ListNode:
 	def __init__(self,val=0,next=None):
 		self.val=val
 		self.next=next

# 在创建空链表的时候只需要将相应的链表头节点设置为空
class LinkedList:
	def __init__(self):
		self.head=None
# 2.2 建立一个线性链表
	def create(self,data):
		self.head=ListNode(0)
		cur=self.head
		for i in range(len(data)):
			node=ListNode(data[i])
			cur.next=node
			cur=cur.next
# a=LinkedList()
# a.create([1,2,3,4])
# print(a.head.next.val)
# 
# 求一个线性链表的长度
	def length(self):
		count=0
		cur=self.head
		while(cur.next!=None):
			count+=1
			cur=cur.next
		return count

	def find(self,val):
		cur=self.head
		while cur.next!=None:
			if val==cur.val:
				return cur
			cur=cur.next
		return None
# 2.5 插入元素
# 链表头部插入元素，在链表的第一个节点之前插入
# 链表尾部插入元素，在链表的最后一个节点之后插入
# 链表中间插入，在链表第i节点插入值节点
	def insert(self,index,value):
		# 新建一个节点
		dummy=ListNode(val)
		if index==0:
			dummy.next=self.head
			self.head=dummy
		else:
			count=0
			cur=self.head
			while(count<index):
				count+=1
				cur=cur.next

			if cur.next:
				dummy.next=cur.next
			cur.next=dummy
# 2.6 改变元素
	def change(self,index,val):
		count=0
		cur=self.head

		while cur and count<idnex:
			count+=1
			cur=cur.next

		if not cur:
			return 'error'

		cur.val=val
# 2.7 删除u元素
# 删除头部
# 删除中间
# 删除尾部


# ===================T707========================# 
class MyLinkedNode:
    def __init__(self,val=0,next=None):
        self.val=val
        self.next=next

class MyLinkedList:

    def __init__(self):
        self.head=None
        
    def get(self, index: int) -> int:
        count=0
        cur=self.head
        while (cur and count<index):
            cur=cur.next
            count+=1
        if cur:
            return cur.val
        else:
            return -1

    def addAtHead(self, val: int) -> None:
        dummy=MyLinkedNode(val)
        dummy.next=self.head
        self.head=dummy


    def addAtTail(self, val: int) -> None:
        dummy=MyLinkedNode(val)
        cur=self.head
        if cur:
        	while cur.next:
	            cur=cur.next
        	cur.next=dummy
        	
        else:
        	self.addAtHead(val)



    def addAtIndex(self, index: int, val: int) -> None:
        if index==0:
            self.addAtHead(val)
        
        else:
 
	        dummy=MyLinkedNode(val)
	        count=0
	        cur=self.head

	        while(cur and count<index-1):
	            count+=1
	            cur=cur.next
	        
	        if cur:
	            if cur.next:
	                dummy.next=cur.next
	                cur.next=dummy
	            else:
	                cur.next=dummy


    def deleteAtIndex(self, index: int) -> None:
        count=0
        cur=self.head
        if index==0:
            self.head=self.head.next
        else:
            while(cur and count<index-1):
                count+=1
                cur=cur.next
            cur.next=cur.next.next

ll=MyLinkedList()
ll.addAtHead(2)
print(ll.get(0))
ll.addAtIndex(0,1)
print(ll.get(1))

# ===================链表排序========================#
# 在数组排序中，常见的排序算法有：冒泡排序、选择排序、插入排序、归并排序、 快速排序、堆排序、位排序
# 桶排序、基数排、希尔排序
# 
# 但是对于链表而言，链表并不支持随机访问，访问链表后面的节点只能依靠next指针从头遍历，因此对于数组排序问题
# 链表排序会更复杂一点
# 
# 重点链表插入排序和链表的归并排序
# ===================链表排序========================#
# 
# 2.1 链表的冒泡排序
# node_i\node_j\tail
def linkkedListSelectionList(head):
	node_i=head
	min_node=node_i
	node_j=node_i.next
	while node_j:
		if node_j.val<min_node.val:
			min_node=node_j
		node_j=node_j.next

	if node_i !=min_node:
		node_i.val,min_node.val=min_node.val,node_i.value
	node_i=node_i,next

	return head
