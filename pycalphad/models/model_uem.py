"""
UEM (Unified Excess Model) 热力学模型实现
稳定化版本 - 平衡功能和稳定性
"""
from pycalphad import Model
from pycalphad import variables as v
from pycalphad.core.utils import wrap_symbol
from sympy import S, Float, exp, Add
from tinydb import where
import logging
import  pycalphad.models.uem_symbolic as uem

# 配置日志
logger = logging.getLogger(__name__)

class ModelUEM(Model):
    """
    PyCalphad-compatible UEM (Unified Excess Model)
    
    UEM模型基于二元系统边界性质的外推来计算多元系统的过剩Gibbs能。
    该模型通过有效摩尔分数的概念来捕捉二元边界之间的相互作用。
    
    Parameters
    ----------
    dbe : Database
        包含相关参数的数据库
    comps : list
        要考虑的组分名称列表
    phase_name : str
        相模型名称
    parameters : dict or list, optional
        要在模型中替换的参数的可选字典
    
    Attributes
    ----------
    components : set
        活性组分集合
    constituents : list
        包含每个亚晶格上组分集合的列表
    """
    # 定义能量贡献项
    contributions = [
        ('ref', 'reference_energy'),
        ('idmix', 'ideal_mixing_energy'),
        ('xsmix', 'excess_mixing_energy')
    ]
    
    
    def excess_mixing_energy(self, dbe):
        comps = [str(c) for c in self.components if str(c) != 'VA']
        expr = uem.get_uem1_excess_gibbs_expr(dbe, comps, self.phase_name, v.T)/self._site_ratio_normalization
        return  expr
    
    def reference_energy(self, dbe):
        """从父类继承引用能量贡献"""
        return super().reference_energy(dbe)
    
    def ideal_mixing_energy(self, dbe):
        """从父类继承理想混合能贡献"""
        return super().ideal_mixing_energy(dbe)


class DummyModel(Model):
    """
    简单的测试模型，用于验证框架
    """
    contributions = [
        ('ref', 'reference_energy'),
        ('idmix', 'ideal_mixing_energy'),
        ('xsmix', 'excess_mixing_energy')
    ]

    def excess_mixing_energy(self, dbe):
        """简单的过剩能测试函数"""
        logger.info("使用DummyModel")
        comps = [str(c) for c in self.components if str(c) != 'VA']
        
        if len(comps) < 2:
            return Float(0.0)
        
        # 简单的二元相互作用
        if 'AL' in comps and 'NI' in comps:
            return v.X('AL') * v.X('NI') * Float(-10000.0)
        
        return Float(0.0)