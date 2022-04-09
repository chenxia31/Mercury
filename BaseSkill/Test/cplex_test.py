# 导入库
from docplex.mp.model import Model
# 创建模型
model = Model()
# 创建变量列表
X = model.continuous_var_list([i for i in range(0, 2)], lb=0, name='X')
# 设定目标函数
model.minimize(2 * X[0] + 3* X[1])
# 添加约束条件
model.add_constraint(3 * X[0] + X[1] >= 30)
model.add_constraint(X[0] - X[1] <= 10)
model.add_constraint(X[1] >= 1)
# 求解模型
sol = model.solve()
# 打印结果
print(sol)
# 打印详细结果
print(sol.solve_details)
