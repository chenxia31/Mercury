class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n==1:
            return ['()']
        else:
            temps=self.generateParenthesis(n-1)
            result=[]
            for temp in temps:
                for j in range(len(temp)+2):
                    result.append(temp[0:j]+'()'+temp[j:])
            return list(set(result))