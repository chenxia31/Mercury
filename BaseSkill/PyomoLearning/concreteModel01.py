# 学习promo中建模的基本过程
# min 2*x+3*y
# s.t. 3x+4y>=1
# x,y>=0
# 尝试构建模型，求解并得到最终的结果
from pyomo.environ import *
import pyomo.environ as pyo

# 建立实际模型
model=ConcreteModel()
model.x=Var([1,2],domain=pyo.NonNegativeReals)
model.Obj=Objective(expr=2*model.x[1]+3*model.x[2])
model.Constraint1=Constraint(expr=3*model.x[1]+4*model.x[2]>=1)

opt=pyo.SolverFactory('cplex')
opt.solve(model)
print(pyo.value(model.x[1]))
print(pyo.value(model.x[2]))