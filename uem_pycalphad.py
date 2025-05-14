# 极简测试脚本
from pycalphad import Database, calculate, variables as v
from pycalphad.models.model_uem import ModelUEM
import logging
import sys

print(sys.path)


# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载数据库
dbf = Database('examples/alcrni.tdb')

# 只测试二元系统
binary_comps = ['AL', 'NI']
binary_phases = ['LIQUID']

# 只测试单个点
conds = {
    v.T: 1800,
    v.P: 101325,
    v.X('AL'): 0.2,
    v.X('NI'): 0.5,
    v.X('CR'): 0.3
}
comps = ['AL', 'NI', 'CR', 'VA']
phases = ['LIQUID']

result = calculate(dbf, comps, phases, model=ModelUEM, conditions=conds)

print("UEM计算结果:", result.GM.values)
result_rkm = calculate(dbf, comps, phases, model=None, conditions=conds)
print("R-K-M计算结果:", result_rkm.GM.values)

