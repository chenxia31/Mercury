class Solution:
    def threeSum(self, nums) :
        # 找出所有和为0的不重复的三元组
        # 数组、双指针和排序
        # 注意答案中不可以包含重复的三元
        n=len(nums)
        nums.sort()
        ans=list()

        for first in range(n):
            if first>0 and nums[first]==nums[first-1]:
                continue
            third=n-1
            target=-nums[first]

            for second in range(first+1,n):
                if second >first +1 and nums[second]==nums[second-1]:
                    continue
                while second < third and nums[second]+nums[third] > target:
                    third-=1

                if second == third:
                    break
                if nums[second]+nums[third]==target:
                    ans.append([nums[first],nums[second],nums[third]])

        return ans



s=Solution()
print(s.threeSum([-1,0,1,2,-1,-4]))