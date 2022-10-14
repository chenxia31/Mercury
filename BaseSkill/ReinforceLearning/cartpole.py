import gym
# 导入gym的python接口环境包
import matplotlib.pyplot as plt
import time
#env for environment，用来构建实验环境
env=gym.make('CartPole-v1')


#重置一个回合
env.reset()

for _ in range(1000):
    # 显示图形界面
    # env.render()
    env.render()
    # 从动作空间中随机选取一个动作
    action=env.action_space.sample()
    # 提交动作，并反馈对应的参数 observation、reward、done、info
    result=env.step(action)
    print(result)
    time.sleep(1)
env.close()