class Solution:
    def minNumber(self, nums) :
        # 参考别人的题解，发现这个是一道排序的题目
        # 排序的方式并不是，A<B那么A就在B前面
        # 而是str（A+B）<str(B+A)那么A就在B前面
        if len(nums)<2:
            return nums
        else:
            mid=len(nums)//2
            left=nums[:mid]
            right=nums[mid:]
            res_arr=self.merge(self.minNumber(left),self.minNumber(right))
            return res_arr
    
    def merge(self,left,right):
        arr=[]
        while left and right:
            if self.compare(left[0],right[0])==1:
                arr.append(left[0])
                left.pop(0)
            else:
                arr.append(right[0])
                right.pop(0)
        
        while left:
            arr.append(left[0])
            left.pop(0)
        
        while right:
            arr.append(right[0])
            right.pop(0)
        
        return arr

    def compare(self,int1,int2):
        s1=str(int1)+str(int2)
        s2=str(int2)+str(int1)
        if int(s1)>int(s2):
            return 0
        else:
            return 1   
print(Solution().minNumber([12,2,4]))
a=[1,2,3]
print(a[-2])