class Solution:
    def longestPalindrome(self, s: str) -> str:
        ## dynamic programing ,基本步骤是确定下标的含义和递推公式的含义
        # 第一步，确定dp数据，用dp【i,j】来表示区间范围内的淄川是否为回文子串dp[i][j]是否为true
        # 第二步，确定递推的三种情况，回文数取真的情况分为
        # 1. i=j相等，true
        # 2. i和j相差等于1，true
        # 3. i和j相差大于1的情况，如果i和j相等，需要查看i和j的区间内部是不是相等，也就是dp[i+1][j-1]是不是true
        # 第三步，确定dp数组如何初始化，当然都是false
        # 第四步，为了保证dp[i+1][j-1]要最新开始计算,所以可以先将第一种情况和第二种情况先得到

        dp=[[0 for i in range(len(s))] for j in range(len(s))]
        left = 0
        right = 0
        maxlength=0
        for i in range(len(s)-1,-1,-1):
            for j in range(i, len(s)):
                if s[i]==s[j]:
                    if (j - i <= 1):
                        dp[i][j] = 1
                    elif dp[i + 1][j - 1] == 1:
                        dp[i][j] = 1
                if dp[i][j] == 1 and j - i + 1 > maxlength:
                    maxlength = j - i + 1
                    left = i
                    right = j
        return s[left:right + 1]

sl=Solution()
print(sl.longestPalindrome('abbacssdddddddddd'))