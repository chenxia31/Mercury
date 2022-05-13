class Solution:
    def maxArea(self, height):
        # 找出其中的两条线，找出其中与x轴构成可以容纳最多的水
        left=0
        right=len(height)-1
        result=0
        while left < right:
            result=max(result,min(height[right],height[left])*(right-left))
            # 更新right和left
            # 这里的目的是消除最大值的可能性
            if height[right]>height[left]:
                left=left+1
            else:
                right=right-1
        return result

s=Solution()
print(s.maxArea([1,2,4,3]))