#%%
class Solution:
    def lengthOfLongestSubstring(self, s):
        length=[]
        for i in range(len(s)):
            for j in range(i+1,len(s)):
                if s[j] in s[i:j]:
                    length.append(j-i)
                    break
        return max(length)


class Solution2:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 判断s为空
        if not s:return 0

        # 设置左指针的位置
        left = 0
        # 设置查找的字符
        lookup = set()
        # 长度
        n = len(s)

        # 最长的长度
        max_len = 0
        # 现有的长度
        cur_len = 0

        # 遍历每一个字符
        for i in range(n):
            # 首先加一
            cur_len += 1
            # 如果字符在里面，就删除这个字符
            while s[i] in lookup:
                lookup.remove(s[left])
                # 左变要转移
                left += 1
                # 现有长度要减少
                cur_len -= 1
            # 根据长度来计算最长长度
            if cur_len > max_len:max_len = cur_len

            # 最初要初始化一个值
            lookup.add(s[i])
        return max_len