

# 求解[1,2,3]的全排列问题
from turtle import back


def premute(nums):
    res=[] # 存放所有符合条件结果的集合
    path=[] # 存放当前符合条件的结果
    def backtracking(nums):
        if len(path)==len(nums):
            res.append(path[:])
            return
        
        for i in range(len(nums)):
            if nums[i] not in path:
                # 从当前路径中没有出现的数字中选择
                path.append(nums[i])
                # 递归搜索
                backtracking(nums)
                # 撤销选择
                path.pop()
        
    backtracking(nums)

    return res
# print(premute([1,2,3,4]))

def solveNQueen(n):
    res=[]
    path=[]

    def backtracking(n):
        if len(path)==n:
            res.append(path)
            return 
        for j in range(n):
            forbid=[]
            for i in range(len(path)):
                forbid.append(path[i])
                forbid.append(path[i]+len(path)-j)
        if i not in forbid:
            path.append(i)
            backtracking(n)

def permuteUnique(nums):
    # 维护一个hashmap
    hashmap={}
    for num in nums:
        if num in hashmap:
            hashmap[num]+=1
        else:
            hashmap[num]=1
    res=[]
    path=[]

    def backtracking(nums):
        if len(path)==len(nums):
            res.append(path[:])
            return 
        
        for i in range(len(nums)):
            print(hashmap)
            while(hashmap[nums[i]]==0):
                i=i+1
            path.append(nums[i])
            hashmap[nums[i]]-=1
            backtracking(nums)
            num=path.pop()
            if num in hashmap:
                hashmap[num]+=1
            else:
                hashmap[num]=1
    backtracking(nums)
    return res
nums = [1,1,2]
# permuteUnique(nums)

# LC22 括号生成
def generateParentthesis(n):
    ans=[]
    path=[]
    def backtrack(S,left,right):
        print(S)
        if len(S)==2*n:
            ans.append(''.join(S))
            return 
        if left<n:
            S.append('(')
            backtrack(S,left+1,right)
            S.pop()
        if right<left:
            S.append(')')
            backtrack(S,left,right+1)
            S.pop()
        
    backtrack([],0,0)
    return ans
# print(generateParentthesis(3))

def letterCombinations(digits):
    num2letter=['abc','def','ghi','jkl','mno','pqrs','yuv','wxyz']
    n=len(digits)
    res=[]
    def backtracking(S):
        if len(S)==n:
            res.append(''.join(S))
            return
        
        index=len(S)
        num=int(digits[index])
        print(num)
        for i in num2letter[num-2]:
            S.append(i)
            backtracking(S)
            S.pop()
    backtracking([])
    return res
# print(letterCombinations('23'))

# LC784 字母大小写排列
def lettercase(s=''):
    res=[]
    path=[]


    def backtracking(path,index):
        if index==len(s):
            # 所有的字母都遍历完毕
            res.append(''.join(path[:]))
            return

        char=s[index]
        if char.isdigit():
            # 如果是数字，直接加上去就行
            path.append(char)
            backtracking(path,index+1)
            path.pop()
        else:
            path.append(char.lower())
            backtracking(path,index+1)
            path.pop()
            path.append(char.upper())
            backtracking(path,index+1)
            path.pop()
    backtracking([],0)
    return res
# print(lettercase('A1b2'))

#LC79 单词搜索
def exist(board, word):
    m,n=len(board),len(board[0])
    before=[-2,-2]
    path=[]
    res=0
    def backtracing(board,i,j):
        nonlocal path
        nonlocal before
        nonlocal res
        # 表示从位置（i，j）开始搜索，来看和word是否相同
        print(len(path))
        print(len(word))
        if len(path)==len(word):
            res=1
            return True
        path.append([i,j])
        print(path)
        if 0<=i<m and 0<=j<n and board[i][j]==word[len(path)-1]:
            for p in path:
                if i==p[0] and j==p[1]:
                    path.pop()
            backtracing(board,i+1,j)
            backtracing(board,i-1,j)
            backtracing(board,i,j+1)
            backtracing(board,i,j-1)
        else:
            path.pop()
            return False

    for k in range(m):
        for l in range(n):
            if board[k][l]==word[0] and backtracing(board,k,l):
                return True
            if path:
                path.pop()
    return res==1

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCB"
print(exist(board,word))