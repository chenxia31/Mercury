# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 题目debug地址：https://leetcode-cn.com/problems/binary-tree-inorder-traversal/

class Solution:
    def inorderTraversal(self, root):
        '''
        题目描述：给定一个二叉树的根节点root，返回它的中序遍历
        二叉树的遍历分为深度遍历和广度遍历
        深度遍历分为前序、中序和后序三种遍历方法
        - 前序，根节点、左子树、右子树
        - 中序，左子树、根节点、右子树
        - 后序，左子树、右子树、根节点
        广度遍历就是常说的层次遍历的方法


        :param root:
        :return:
        '''
        stack,ret=[],[]
        # 设计一个栈作为中间变量
        # ret为最终结果
        cur=root # 初始化最初的值为cur
        while stack or cur:
            # 当stack为空并且cur为空停止，也就是所有都遍历完，同时也不存在左子树
            if cur:
                # 如果当前节点非空，则取这个节点到stack中
                # 将当前节点设置为节点的left node
                stack.append(cur)
                cur=cur.left
            else:
                # 如果stack非空，但是当前节点为空。说明遍历到最左边的
                # 取当前节点的上一个
                cur=stack.pop()
                ret.append(cur.val)
                cur=cur.right
        return ret
