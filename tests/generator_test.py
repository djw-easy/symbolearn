"""
复杂度约束表达式生成器使用示例
"""
import numpy as np
from src.generator import ExprGenerator
from src.constraints import ComplexityConstraints


def example_1_basic_usage():
    """示例1：基本使用（不使用复杂度约束）"""
    print("=" * 80)
    print("示例1：基本使用（原始功能）")
    print("=" * 80)
    
    generator = ExprGenerator(
        maxsize=20,
        operators=['+', '-', '*', '/', 'sin', 'cos', 'exp'],
        n_variables=5,
        use_constants=True,
        use_variables=True,
        random_state=42
        # 不传入任何复杂度参数，自动使用原始方法
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    
    # 生成几个表达式
    for i in range(3):
        expr = generator.generate_random_expr()
        print(f"表达式 {i+1}: size={expr.tree.size}")
    
    print()


def example_2_operator_complexity():
    """示例2：操作符复杂度约束"""
    print("=" * 80)
    print("示例2：操作符复杂度约束")
    print("=" * 80)
    
    # 定义操作符复杂度
    complexity_of_operators = {
        'sin': 2,
        'cos': 2,
        'exp': 3,
        '+': 1,
        '-': 1,
        '*': 1,
        '/': 1
    }
    
    generator = ExprGenerator(
        maxsize=10,  # 最大复杂度为10
        operators=['+', '-', '*', '/', 'sin', 'cos', 'exp'],
        n_variables=3,
        use_constants=True,
        use_variables=True,
        random_state=42,
        # 传入操作符复杂度，自动启用复杂度约束
        complexity_of_operators=complexity_of_operators,
        complexity_of_constants=1.0,
        complexity_of_variables=1.0
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    print(f"原因: 操作符复杂度不全为1 (sin=2, cos=2, exp=3)")
    print()
    
    # 生成表达式
    for i in range(3):
        expr = generator.generate_random_expr(target_complexity=8.0)
        print(f"表达式 {i+1}: 目标复杂度=8.0")
    
    print()


def example_3_variable_complexity():
    """示例3：变量复杂度约束"""
    print("=" * 80)
    print("示例3：不同变量不同复杂度")
    print("=" * 80)
    
    # 定义变量复杂度（x0,x1简单，x2,x3中等，x4复杂）
    complexity_of_variables = [1.0, 1.0, 2.0, 2.0, 3.0]
    
    generator = ExprGenerator(
        maxsize=15,
        operators=['+', '-', '*', '/'],
        n_variables=5,
        use_constants=True,
        use_variables=True,
        random_state=42,
        # 传入变量复杂度，自动启用复杂度约束
        complexity_of_operators={'+': 1, '-': 1, '*': 1, '/': 1},
        complexity_of_constants=1.5,  # 常量复杂度不为1
        complexity_of_variables=complexity_of_variables
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    print(f"原因: 常量复杂度={generator.complexity_constraints.complexity_of_constants}, 变量复杂度不全为1")
    print()
    
    # 生成表达式
    expr = generator.generate_random_expr(target_complexity=10.0)
    print(f"生成的表达式（目标复杂度=10.0）")
    print(f"说明：x0,x1 复杂度为1，x2,x3 复杂度为2，x4 复杂度为3")
    
    print()


def example_4_parameter_constraints():
    """示例4：参数复杂度约束"""
    print("=" * 80)
    print("示例4：参数复杂度约束")
    print("=" * 80)
    
    # 定义参数约束
    constraints = {
        'sin': 5,           # sin的参数复杂度不超过5
        'pow': (-1, 1),     # 左参数无限制，右参数复杂度≤1
        '/': (10, 3),       # 分子≤10，分母≤3
        '*': 8              # 两个参数复杂度都不超过8
    }
    
    generator = ExprGenerator(
        maxsize=20,
        operators=['+', '-', '*', '/', 'sin', 'pow'],
        n_variables=3,
        use_constants=True,
        use_variables=True,
        random_state=42,
        # 传入参数约束，自动启用复杂度约束
        complexity_of_operators={'sin': 2, 'pow': 2, '+': 1, '-': 1, '*': 1, '/': 1},
        complexity_of_constants=1.0,
        complexity_of_variables=1.0,
        constraints=constraints
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    print(f"原因: 存在参数约束")
    print()
    
    print("约束说明:")
    print("  - sin的参数复杂度 ≤ 5")
    print("  - pow(左参数无限制, 右参数≤1)")
    print("  - /(分子≤10, 分母≤3)")
    print("  - *(两个参数都≤8)")
    print()
    
    # 生成表达式
    expr = generator.generate_random_expr(target_complexity=15.0)
    print(f"生成的表达式（目标复杂度=15.0）")
    print("✓ 允许: pow(x0 + x1, 2) （左=3，右=1）")
    print("✗ 不允许: pow(x0, sin(x1)) （右=3，超过限制1）")
    
    print()


def example_5_nested_constraints():
    """示例5：嵌套层数约束"""
    print("=" * 80)
    print("示例5：嵌套层数约束")
    print("=" * 80)
    
    # 定义嵌套约束
    nested_constraints = {
        'sin': {
            'cos': 0,   # cos不能出现在sin内部
            'sin': -1   # sin可以无限嵌套在sin内部（-1=无限制）
        },
        'cos': {
            'cos': 2,   # cos最多嵌套2层在cos内部
            'sin': 1    # sin最多嵌套1层在cos内部
        }
    }
    
    generator = ExprGenerator(
        maxsize=20,
        operators=['+', '-', 'sin', 'cos'],
        n_variables=3,
        use_constants=True,
        use_variables=True,
        random_state=42,
        # 传入嵌套约束，自动启用复杂度约束
        complexity_of_operators={'sin': 2, 'cos': 2, '+': 1, '-': 1},
        complexity_of_constants=1.0,
        complexity_of_variables=1.0,
        nested_constraints=nested_constraints
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    print(f"原因: 存在嵌套约束")
    print()
    
    print("约束说明:")
    print("  - sin内部:")
    print("    • cos不能出现（最大嵌套0层）")
    print("    • sin可以无限嵌套")
    print("  - cos内部:")
    print("    • cos最多嵌套2层")
    print("    • sin最多嵌套1层")
    print()
    
    print("示例:")
    print("  ✓ 允许: sin(sin(sin(x0))) （sin无限嵌套）")
    print("  ✗ 不允许: sin(cos(x0)) （cos不能在sin内部）")
    print("  ✓ 允许: cos(cos(cos(x0))) （嵌套2层）")
    print("  ✗ 不允许: cos(cos(cos(cos(x0)))) （嵌套3层，超过限制）")
    
    print()


def example_6_comprehensive():
    """示例6：综合使用所有约束"""
    print("=" * 80)
    print("示例6：综合使用所有约束")
    print("=" * 80)
    
    # 操作符复杂度
    complexity_of_operators = {
        'sin': 2,
        'cos': 2,
        'exp': 3,
        '+': 1,
        '-': 1,
        '*': 1.5,
        '/': 1.5
    }
    
    # 变量复杂度
    complexity_of_variables = [1.0, 1.0, 2.0, 2.0, 3.0]
    
    # 参数约束
    param_constraints = {
        'sin': 5,
        'exp': 4,
        '/': (10, 3),
        '*': 8
    }
    
    # 嵌套约束
    nested_constraints = {
        'exp': {
            'exp': 0,  # exp不能嵌套在exp内部
            'sin': 1,  # sin最多嵌套1层在exp内部
            'cos': 1
        },
        'sin': {
            'cos': 0
        }
    }
    
    generator = ExprGenerator(
        maxsize=25,
        operators=['+', '-', '*', '/', 'sin', 'cos', 'exp'],
        n_variables=5,
        variable_names=['x', 'y', 'z', 'u', 'v'],
        use_constants=True,
        use_variables=True,
        random_state=42,
        # 自动启用复杂度约束（多个触发条件）
        complexity_of_operators=complexity_of_operators,
        complexity_of_constants=1.5,
        complexity_of_variables=complexity_of_variables,
        complexity_of_aggregations=2.0,
        constraints=param_constraints,
        nested_constraints=nested_constraints
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    print(f"原因: 存在多个触发条件（操作符复杂度、变量复杂度、常量复杂度、参数约束、嵌套约束）")
    print()
    
    print("配置说明:")
    print("1. 操作符复杂度:")
    print("   sin=2, cos=2, exp=3, +=-=1, *=/=1.5")
    print()
    print("2. 变量复杂度:")
    print("   x=1, y=1, z=2, u=2, v=3")
    print()
    print("3. 常量和聚合复杂度:")
    print("   常量=1.5, 聚合=2.0")
    print()
    print("4. 参数约束:")
    print("   sin(参数≤5), exp(参数≤4), /(分子≤10,分母≤3), *(参数≤8)")
    print()
    print("5. 嵌套约束:")
    print("   exp内: 不允许exp，sin/cos最多1层")
    print("   sin内: 不允许cos")
    print()
    
    # 生成几个表达式
    for i in range(3):
        expr = generator.generate_random_expr(target_complexity=20.0)
        print(f"表达式 {i+1}: 目标复杂度=20.0")
    
    print()


def example_7_aggregation_complexity():
    """示例7：聚合操作的复杂度约束"""
    print("=" * 80)
    print("示例7：聚合操作的复杂度约束")
    print("=" * 80)
    
    generator = ExprGenerator(
        maxsize=15,
        operators=['+', '-', '*', '/'],
        n_variables=10,
        use_constants=True,
        use_variables=True,
        use_aggregations=True,
        aggregation_operators=['mean', 'max', 'min'],
        random_state=42,
        # 聚合复杂度不为1，自动启用复杂度约束
        complexity_of_operators={'+': 1, '-': 1, '*': 1, '/': 1},
        complexity_of_constants=1.0,
        complexity_of_variables=1.0,
        complexity_of_aggregations=2.5  # 聚合操作复杂度为2.5
    )
    
    print(f"自动判断: use_complexity_constraints={generator.use_complexity_constraints}")
    print(f"原因: 聚合复杂度={generator.complexity_constraints.complexity_of_aggregations} ≠ 1")
    print()
    
    print("配置说明:")
    print("  - 变量复杂度: 1.0")
    print("  - 常量复杂度: 1.0")
    print("  - 聚合复杂度: 2.5")
    print("  - 聚合操作示例: mean(v1-v5), max(v2-v8)")
    print()
    
    # 生成表达式
    expr = generator.generate_random_expr(target_complexity=12.0)
    print(f"生成的表达式（目标复杂度=12.0）")
    
    print()


def example_8_comparison():
    """示例8：对比有无复杂度约束的区别"""
    print("=" * 80)
    print("示例8：对比有无复杂度约束")
    print("=" * 80)
    
    # 不使用复杂度约束（所有复杂度都是1）
    print("A. 不使用复杂度约束（原始方法）:")
    generator_basic = ExprGenerator(
        maxsize=15,
        operators=['+', '-', 'sin', 'exp'],
        n_variables=3,
        random_state=42
        # 不传入任何复杂度参数
    )
    
    print(f"   自动判断: use_complexity_constraints={generator_basic.use_complexity_constraints}")
    
    for i in range(2):
        expr = generator_basic.generate_random_expr()
        print(f"  表达式 {i+1}: size={expr.tree.size}")
    
    print()
    
    # 使用复杂度约束
    print("B. 使用复杂度约束:")
    generator_complex = ExprGenerator(
        maxsize=15,
        operators=['+', '-', 'sin', 'exp'],
        n_variables=3,
        random_state=42,
        # 传入复杂度参数，自动启用
        complexity_of_operators={'sin': 2, 'exp': 3, '+': 1, '-': 1},
        complexity_of_constants=1.0,
        complexity_of_variables=1.0,
        constraints={'sin': 5, 'exp': 4}
    )
    
    print(f"   自动判断: use_complexity_constraints={generator_complex.use_complexity_constraints}")
    print(f"   原因: 操作符复杂度不全为1, 存在参数约束")
    
    for i in range(2):
        expr = generator_complex.generate_random_expr(target_complexity=10.0)
        print(f"  表达式 {i+1}: 目标复杂度=10.0")
    
    print()
    print("说明:")
    print("  - 原始方法：基于节点数量限制")
    print("  - 复杂度约束：基于加权复杂度限制，可以更精细地控制表达式结构")
    print()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("*" * 80)
    print("复杂度约束表达式生成器 - 使用示例")
    print("*" * 80)
    print("\n")
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_operator_complexity()
    example_3_variable_complexity()
    example_4_parameter_constraints()
    example_5_nested_constraints()
    example_6_comprehensive()
    example_7_aggregation_complexity()
    example_8_comparison()
    
    print("*" * 80)
    print("所有示例运行完成！")
    print("*" * 80)


if __name__ == "__main__":
    main()