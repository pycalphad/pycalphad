"""
UEM1 (Unified Excess Model 1) 符号计算模块
"""
from sympy import Symbol, Add, Mul, Pow, Abs, exp, simplify, S, Piecewise, nan, Basic
from pycalphad import variables as v
from pycalphad.core.utils import wrap_symbol
from pycalphad.variables import R
from tinydb import where
import logging

# 配置日志
logger = logging.getLogger(__name__)


def is_binary_in_phase (arr, comp1, comp2):
	"""
	检查成分数组是否只包含指定的两个组分

	Parameters
	----------
	arr : list
		相组成的二维数组
	comp1 : str
		第一个组分名称
	comp2 : str
		第二个组分名称

	Returns
	-------
	bool
		如果数组只包含两个指定组分则返回True
	"""
	comps = set(str(s) for subl in arr for s in subl)
	return comp1 in comps and comp2 in comps and len(comps) == 2


_param_cache = {}

def wrap_parameter_safely(parameter):
	
	return parameter


def uem1_delta_expr (dbe, comp1, comp2, phase_name, T):
	"""
	计算UEM1模型中两个组分之间的性质差表达式，非交互条件下

	Parameters
	----------
	dbe : Database
		包含相关参数的数据库
	comp1 : str
		第一个组分名称
	comp2 : str
		第二个组分名称
	phase_name : str
		相名称
	T : StateVariable
		温度变量

	Returns
	-------
	SymPy表达式
		归一化后的delta表达式
	"""
	
	x = Symbol('x')
	G_ex = S.Zero
	
	param_query = (
			(where('phase_name') == phase_name) &
			(where('parameter_type') == 'G') &
			(where('constituent_array').test(lambda arr: is_binary_in_phase(arr, comp1, comp2)))
	)
	params = dbe.search(param_query)
	
	if not params:
		logger.warning(f"没有找到{comp1}-{comp2}的参数")
		return S.Zero
	
	
	for p in params:
		try:
			order = p['parameter_order']
			param = wrap_parameter_safely(p['parameter'])
			G_ex += Mul(x, 1 - x, param, Pow(2 * x - 1, order))
		except Exception as e:
			logger.warning(f"处理参数时出错: {str(e)}")
			continue
	
	dGdx = G_ex.diff(x).subs(x, 0)
	dGdx_sym = G_ex.subs(x, 1 - x).diff(x).subs(x, 0)
	
	# 计算绝对差值并除以R*T
	delta = Abs(dGdx - dGdx_sym) / (R * T)
	normalized_delta = simplify(delta)
	
	return normalized_delta


def uem1_contribution_ratio (dbe, k, i, j, phase_name, T):
	"""
	计算组分k对i-j二元对的贡献系数

	Parameters
	----------
	dbe : Database
		包含相关参数的数据库
	k : str
		第三组分名称
	i : str
		第一个组分名称
	j : str
		第二个组分名称
	phase_name : str
		相名称
	T : StateVariable
		温度变量

	Returns
	-------
	SymPy表达式
		贡献比率
	"""
	delta_ki = uem1_delta_expr(dbe, k, i, phase_name, T)
	delta_kj = uem1_delta_expr(dbe, k, j, phase_name, T)
	
	if delta_ki == S.Zero and delta_kj == S.Zero:
		return S.Half
	result = simplify((delta_kj / (delta_ki + delta_kj)) * exp(-delta_ki))
	print('贡献系数：'+k + 'to'+i+ 'in\t'+ i +'-'+ j+ str(result.evalf(subs={T:1800.0})))
	return result


def construct_binary_excess (dbe, comp_i, comp_j, phase_name, x_eff_i, x_eff_j):
	"""
	构造二元过剩能表达式

	Parameters
	----------
	dbe : Database
		包含相关参数的数据库
	comp_i : str
		第一个组分名称
	comp_j : str
		第二个组分名称
	phase_name : str
		相名称
	x_eff_i : SymPy表达式
		第一个组分的有效摩尔分数
	x_eff_j : SymPy表达式
		第二个组分的有效摩尔分数

	Returns
	-------
	SymPy表达式
		二元过剩能表达式
	"""
	G_ex_ij = S.Zero
	param_query = (
			(where('phase_name') == phase_name) &
			(where('parameter_type') == 'G') &
			(where('constituent_array').test(lambda arr: is_binary_in_phase(arr, comp_i, comp_j)))
	)
	params = dbe.search(param_query)
	
	for p in params:
		try:
			order = p['parameter_order']
			param = wrap_parameter_safely(p['parameter'])
			term = Mul(x_eff_i, x_eff_j, param, Pow(x_eff_i - x_eff_j, order))
			G_ex_ij = Add(G_ex_ij, term)
		except Exception as e:
			logger.warning(f"处理参数时出错: {str(e)}")
			continue
	
	return G_ex_ij


def is_stable_expression (expr):
	"""检查表达式是否数值稳定"""
	from sympy import zoo, oo, nan, preorder_traversal
	
	# 检查无穷大和NaN
	if expr.has(zoo) or expr.has(oo) or expr.has(nan):
		return False
	
	# 检查除以非常小的数
	for node in preorder_traversal(expr):
		if node.is_Pow and node.args[1] < 0:  # 负幂表示除法
			return False
		if node.is_Mul:
			for arg in node.args:
				if arg.is_Pow and arg.args[1] < 0:
					return False
	
	return True


def get_uem1_excess_gibbs_expr (dbe, comps, phase_name, T):
	"""
	构建UEM1过剩Gibbs能表达式

	Parameters
	----------
	dbe : Database
		包含相关参数的数据库
	comps : list
		组分列表
	phase_name : str
		相名称
	T : StateVariable
		温度变量

	Returns
	-------
	SymPy表达式
		UEM1过剩Gibbs能表达式
	"""
	
	x = {comp: v.X(comp) for comp in comps}
	expr_list = []
	
	for i_idx in range(len(comps)):
		for j_idx in range(i_idx + 1, len(comps)):
			comp_i = comps[i_idx]
			comp_j = comps[j_idx]
			
			# 有效摩尔量计算
			x_eff_i = x[comp_i]
			x_eff_j = x[comp_j]
			
			for k in comps:
				if k not in [comp_i, comp_j]:
					try:
						r_ki = uem1_contribution_ratio(dbe, k, comp_i, comp_j, phase_name, T)
						r_kj = uem1_contribution_ratio(dbe, k, comp_j, comp_i, phase_name, T)
						x_eff_i += r_ki * x[k]
						x_eff_j += r_kj * x[k]
					
					except Exception as e:
						logger.warning(f"计算贡献比率时出错: {str(e)}")
						continue
			
			denominator = x_eff_i + x_eff_j
			
			Xi_ij = x_eff_i / denominator
			Xj_ij = x_eff_j / denominator
			G_ex_ij = construct_binary_excess(dbe, comp_i, comp_j, phase_name, Xi_ij, Xj_ij)
			G_ex_ij = simplify(G_ex_ij.subs({S.NaN: S.Zero}))
			
			# 构造权重
			if Xi_ij == S.Zero or Xj_ij == S.Zero:
				ratio = S.Zero
			else:
				ratio = (x[comp_i] * x[comp_j]) / (Xi_ij * Xj_ij)
			expr_list.append(G_ex_ij * ratio)
	
	total_expr = Add(*expr_list)
	
	total_expr = total_expr.subs(nan, 0)
	
	return total_expr
