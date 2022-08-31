class Solution:
    def removeDuplicates(self, nums) -> int:
        # slow=1
        # fast=1
        # while fast<len(nums):
        #     if nums[fast]!=nums[fast-1]:
        #         nums[slow]=nums[fast]
        #         slow+=1
        #     fast+=1
        # return slow
        indexs=list(map(lambda x: x[0]-x[1], zip(nums[1:len(nums)],nums[:-1])))
        indexs=[1]+indexs
        nums=[num  for num,index in zip(nums,indexs) if index==1]
        print((nums))
        return sum(indexs)

sl=Solution()
nums=[1,1,2,3,4]
sl.removeDuplicates(nums)
print(nums)