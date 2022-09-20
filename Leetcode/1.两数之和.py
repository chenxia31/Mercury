#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res=dict()
        for i in range(len(nums)):
            res[target-nums[i]]=i
            if nums[i] in res:
                return [i,res[nums[i]]]
# @lc code=end

