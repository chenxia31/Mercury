class Solution:
    def reverse(self, x: int) -> int:
        '''
        给定一个32位的有符号整数x，返回x中的数字部分反转之后的结果，如果反转之后的整数超过32位的有符号整数的范围【-2^31，2^31-1】就返回0
        尤其：环境中不允许出现存储64位整数

        常见的比如126的翻转成为621就会溢出
        :param x: 需要翻转的数目
        :return: 翻转之后的数字

        123->321
        120->21
        '''
        flag=x>0
        x=abs(x)
        b=[]
        while(x>0):
            b.append(int(x%10))
            x=(x-x%10)/10

        # 判断有没有溢出
        max=[[2,1,4,7,4,8,3,6,4,8],[2,1,4,7,4,8,3,6,4,7]]
        maxTarget = max[flag]
        i=0
        result=0
        for j in range(len(b)):
            # 判断有没有溢出
            if len(b)>=10:
                # 位数大于10才有意义
                if b[i]>=maxTarget[i]:
                    if b[i]>maxTarget[i]:
                        return 0
                    else:
                        i=i+1
                # 计算结构
            result=result*10+b[j]

        if flag==True:
            return result
        else:
            return -1*result

a=Solution()
print(a.reverse(123))




