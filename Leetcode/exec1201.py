class Solution:
    def matrixBlockSum(self, mat, k):
        self.matrix=mat
        self.presumMatrix=self.preSum() 
        res=[[0 for _ in range(len(mat))] for _ in range(len(mat[-1]))]
        print(self.presumMatrix)
        for i in range(len(res)):
            for j in range(len(res)):
                row1=max(0,i-k)
                col1=max(0,j-k)
                row2=min(i+k,len(mat)-1)
                col2=min(j+k,len(mat[-1])-1)
                res[i][j]=self.presumMatrix[row2+1][col2+1]+self.presumMatrix[row1][col1]-self.presumMatrix[row2+1][col1]-self.presumMatrix[row1][col2+1]
        return res
        

    def preSum(self):
        m=len(self.matrix)
        n=len(self.matrix[0])
        dp=[[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                dp[i][j]=dp[i][j-1]+dp[i-1][j]-dp[i-1][j-1]+self.matrix[i-1][j-1]
        return dp
mat = [[1,2,3],[4,5,6],[7,8,9]]
k = 1

print(Solution().matrixBlockSum(mat,k))