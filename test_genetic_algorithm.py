import pytest
from genetic_algorithm import evalAdjustment
import numpy as np

def test_evalAdjustment():
    # 定义测试用例的参数
    individual = [0.5, -0.2, 0.1]
    initial_forces = np.array([100, 150, 200])
    target_forces = np.array([105, 147, 205])
    influence_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    max_force = 300
    min_adjustment = -1
    max_adjustment = 1
    tolerance = 0.05

    # 调用evalAdjustment函数
    result = evalAdjustment(individual, initial_forces, target_forces, influence_matrix, max_force, min_adjustment, max_adjustment, tolerance)

    # 断言：检查函数返回的mse加上penalties是否符合预期
    expected = (np.mean(np.square(((initial_forces + np.dot(influence_matrix, individual)) - target_forces) / target_forces)) ,)
    assert result == expected, "evalAdjustment函数的返回值与预期不符"
