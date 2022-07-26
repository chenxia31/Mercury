# 二进制枚举子集
def binaryEnumerating(S):
    sub_sets=[]
    n=len(S)
    # 1<<n 相当于2^n次方，range(1<<n)相当于0～2^n-1
    for i in range(1<<n):
        sub_set=[]
        for j in range(n):
            # &1 相当于取最后一位
            if i>>j&1:
                sub_set.append(S[j])
        sub_sets.append(sub_set)
    return sub_sets

# print(binaryEnumerating([3,4,5]))

# LC-1925 统计平方和三元组的数目
from math import sqrt
def countTriples(n):
    # 方法一
    # count=0
    # for i in range(1,n+1):
    #     for j in range(1,n+1):
    #         for q in range(n,max(i,j),-1):
    #             if q*q==i*i+j*j:
    #                 count+=1
    # return count
    # 方法二
    count=0
    for a in range(1,n+1):
        for b in range(1,n+1):
            c=int(sqrt(a*a+b*b))
            if c<=n and a*a+b*b==c*c:
                count+=1
    return count
# print(countTriples(10))

#LC1450-在既定时间做作业的学生人数
def busyStudent(startTime,endTime,queryTime):
    n=len(startTime)
    cnt=0
    for i in range(n):
        if startTime[i]<=queryTime<=endTime[i]:
            cnt+=1
    return cnt

def findContinuousSequence(target):
    # 方法1 超时
    # sub_sets=[]
    # for i in range(1,1+int(target/2)):
    #     print(i)
    #     for j in range(i+1,2+int(target/2)):
    #         print(j)
    #         sum=(i+j)*(j-i+1)/2
    #         print(sum)
    #         if sum==target:
    #             sub_sets.append(list(range(i,j+1)))
    # return sub_sets
    # 方法2:滑动窗口
    left=1
    right=2
    res=[]
    while left<right:
        sum=(left+right)*(right-left+1)//2

        if sum==target:

            res.append(list(range(left,right+1)))
            left+=1
        elif sum<target:
            right+=1
        elif sum>target:
            left+=1
    return res

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt=0
        left=0
        right=0
        sumt=nums[0]
        while(right<len(nums) and left<=right):
            if sumt==k:
                cnt+=1
                right+=1
                left+=1
                if right==len(nums):
                    break
                sumt+=nums[right]
                sumt-=nums[left-1]
                
            elif sumt<k:
                right+=1
                if right==len(nums):
                    break
                sumt+=nums[right]
            elif sumt>k:
                left+=1
                sumt-=nums[left-1]
        
        return cnt

# print(findContinuousSequence(9))

# LC最大正方形，在0和1组成的二维矩阵中找到只包含1 的最大正方形
def maximalSquare(matrix):
    # 最大面积的正方形
    # 积分图的方式
    
    m,n=len(matrix),len(matrix[0])
    dp=[]
    for i in range(m):
        sub_dp=[]
        for j in range(n):
            matrix[i][j]=int(matrix[i][j])
            sub_dp.append(matrix[i][j])
        dp.append(sub_dp)

    max_len=0      
    for i in range(1,m):
        for j in range(1,n):
            if matrix[i][j]==1 and dp[i-1][j]>=1 and dp[i][j-1]>=1 and dp[i-1][j-1]>=1:
                dp[i][j]=min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1
                if dp[i][j]>max_len:
                    max_len=dp[i][j]

    return max_len
matrix = [["0","0","0","1"],["1","1","0","1"],["1","1","1","1"],["0","1","1","1"],["0","1","1","1"]]
print(maximalSquare(matrix))

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_dic = {0: 1}
        pre_sum = 0
        count = 0
        for num in nums:
            pre_sum += num
            if pre_sum - k in pre_dic:
                count += pre_dic[pre_sum - k]
            if pre_sum in pre_dic:
                pre_dic[pre_sum] += 1
            else:
                pre_dic[pre_sum] = 1
        return count