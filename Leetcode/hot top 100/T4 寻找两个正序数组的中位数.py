
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        # even 需要找（m+n+1）/2
        # odd 需要找（m+n）/2 （m+n）/2+1
        m=len(nums1)
        n=len(nums2)
        if (m+n)%2 ==1:
            # odd

            return self.getK(nums1,nums2,(m+n+1)/2)
        else:

            a=self.getK(nums1,nums2,(m+n)/2)
            b=self.getK(nums1,nums2,(m+n)/2+1)
            return (a+b)/2


    def getK(self,nums1,nums2,k):
        # nums1：第一个正序的数组
        # nums2：第二正序的数组
        # k：两个并列数组中第k大的数组
        # 终止条件为，k为1

        k=int(k)


        m=len(nums1)
        n=len(nums2)


        index1,index2=int(min(k//2,m))-1,int(min(k//2,n))-1
        # 两个正序数组中的值

        if m==0:
            # 第一个数组到头?
            return nums2[k-1]

        if n==0:
            # 说明第二个数组到头
            return nums1[k-1]

        if k==1:
            return min(nums1[0],nums2[0])

        # 下面来看如何更新迭代
        value1=nums1[index1]
        value2=nums2[index2]

        if value1 <= value2:
            k=k-index1-1
            nums1=nums1[index1+1:]
        else:
            k=k-index2-1
            nums2=nums2[index2+1:]
        return self.getK(nums1,nums2,k)

s=Solution()
a=[1,2]
b=[3,4,5,6,7]
print(s.findMedianSortedArrays(a,b))