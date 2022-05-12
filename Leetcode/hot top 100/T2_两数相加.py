class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1, l2):
        def addTwoNumbers2(l1, l2, flag):
            '''
            三种情况
            情况1：链表都存在，l1!=None
            情况2：链表不存在，赋值为0

            计算两数之和
            如果大于10，则val为余数

            flag值为1

            链表不存在并且flag==0，则循环终止，输出最终的链表

            链表存在，则继续下一个循环
            :param l1:
            :param l2:
            :param flag:
            :return:
            '''
            if l1!=None:
                num1=l1.val
                l1=l1.next
            else:
                num1 = 0

            if l2!=None:
                num2=l2.val
                l2=l2.next
            else:
                num2 = 0

            # 更新求和
            sum_result = num1 + num2 + flag

            # 结果求余
            val = sum_result % 10

            # 更新flag，大于9设置为1，其他的则为0
            if sum_result > 9:
                flag = 1
            else:
                flag = 0

            # 终止条件，l1为空，并且flag为0
            if l1 == None and l2==None and  flag == 0:
                return ListNode(val)

            return ListNode(val, addTwoNumbers2(l1, l2, flag))
        return addTwoNumbers2(l1,l2,0)

l1=ListNode(0,ListNode(4,ListNode(3)))
l2=ListNode(1,ListNode(6,ListNode(4)))
s=Solution()
print(s.addTwoNumbers(l1,l2))


