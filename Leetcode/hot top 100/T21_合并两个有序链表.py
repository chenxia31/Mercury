
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1, list2):
        dummy=ListNode(0)

        head=ListNode(0)
        dummy.next = head
        while list1.val!=None and list2.val!=None:
            if list1.val<list2.val:
                head.next=list1
                head=head.next
                list1=list1.next
            else:
                head.next=list2
                head=head.next
                list2=list2.next
        if list1.val==None:
            head.next=list2
        if list2.val==None:
            head.next=list1


        return dummy.next.next