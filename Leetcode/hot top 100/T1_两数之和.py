def twoSum(nums,target):
    '''
    给定一个整数数组nums和一个整数目标target，在数组中找到和为目标值的下标
    '''
    result=[]
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i!=j:
                if nums[i]+nums[j]==target:
                    result.append(i)
                    result.append(j)
                     return result


def twoSum(nums,target):
    '''
    给定一个整数数组nums和一个整数目标target，在数组中找到和为目标值的下标
    '''
    hashtable=dict()
    for i in range(len(nums)):
        if target-nums[i] in hashtable:
            return [i,hashtable[target-nums[i]]
        hashtable[nums[i]]=i
    return []