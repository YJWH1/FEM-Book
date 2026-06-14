import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元长度、方向余弦、全局坐标系6×6刚度矩阵
    :param x1: 节点1坐标 [x1, y1, z1]
    :param x2: 节点2坐标 [x2, y2, z2]
    :param E: 弹性模量 (Pa)
    :param A: 横截面积 (m^2)
    :return: L, (cx, cy, cz), Ke
    """
    # 坐标差
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    # 单元长度
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    # 检查节点重合
    if abs(L) < 1e-12:
        raise ValueError("错误：两个节点坐标重合，无效单元！")

    # 方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 构造6×6全局刚度矩阵
    EA_L = E * A / L
    Ke = EA_L * np.array([
        [cx**2,    cx*cy,    cx*cz,   -cx**2,   -cx*cy,   -cx*cz],
        [cx*cy,    cy**2,    cy*cz,   -cx*cy,   -cy**2,   -cy*cz],
        [cx*cz,    cy*cz,    cz**2,   -cx*cz,   -cy*cz,   -cz**2],
        [-cx**2,  -cx*cy,   -cx*cz,    cx**2,    cx*cy,    cx*cz],
        [-cx*cy,  -cy**2,   -cy*cz,    cx*cy,    cy**2,    cy*cz],
        [-cx*cz,  -cy*cz,   -cz**2,    cx*cz,    cy*cz,    cz**2]
    ])

    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算应变、应力、轴力
    :param x1: 节点1坐标
    :param x2: 节点2坐标
    :param E: 弹性模量 (Pa)
    :param A: 横截面积 (m^2)
    :param de: 节点位移列阵 [u1,v1,w1,u2,v2,w2] (m)
    :return: epsilon(应变), sigma(应力,Pa), N(轴力,N)
    """
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    L = np.sqrt(dx**2 + dy**2 + dz**2)
    if abs(L) < 1e-12:
        raise ValueError("错误：两个节点坐标重合，无效单元！")

    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 应变位移矩阵 B
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L
    epsilon = B @ np.array(de)

    # 应力、轴力
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


def run_example1():
    """算例1：沿X轴一维杆单元"""
    print("=" * 70)
    print("========== 算例1：沿X轴一维杆单元 ==========")
    x1 = [0, 0, 0]
    x2 = [2, 0, 0]
    E = 200e9    # 200 GPa
    A = 1.0e-4   # m^2
    de = [0, 0, 0, 1.0e-3, 0, 0]

    try:
        L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
        cx, cy, cz = dir_cos
        eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)

        print(f"单元长度 L = {L:.4f} m")
        print(f"方向余弦: cx={cx:.4f}, cy={cy:.4f}, cz={cz:.4f}")
        print("\n单元刚度矩阵 Ke (6×6):")
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print(Ke)

        print(f"\n轴向应变 ε = {eps:.6e}")
        print(f"轴向应力 σ = {sig/1e6:.4f} MPa")
        print(f"轴力 N = {N:.2f} N")
    except Exception as e:
        print("运行异常：", e)


def run_example2():
    """算例2：空间任意方向杆单元"""
    print("\n" + "=" * 70)
    print("========== 算例2：空间任意方向杆单元 ==========")
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9    # 210 GPa
    A = 2.0e-4   # m^2
    de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    try:
        L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
        cx, cy, cz = dir_cos
        eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)

        print(f"单元长度 L = {L:.4f} m")
        print(f"方向余弦: cx={cx:.4f}, cy={cy:.4f}, cz={cz:.4f}")
        print("\n单元刚度矩阵 Ke (6×6):")
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print(Ke)

        # 验证对称性
        is_symmetric = np.allclose(Ke, Ke.T)
        print(f"\n1. 刚度矩阵是否对称: {is_symmetric}")

        # 验证刚体位移：整体平移
        de_rigid = np.array([1, 1, 1, 1, 1, 1])
        Fe_rigid = Ke @ de_rigid
        print(f"2. 刚体平移节点力二范数: {np.linalg.norm(Fe_rigid):.2e} (趋近于0)")

        # 特征值验证（半正定、奇异性）
        eig_vals = np.linalg.eigvalsh(Ke)
        print(f"3. 刚度矩阵所有特征值:\n{np.round(eig_vals, 4)}")
        zero_count = np.sum(np.abs(eig_vals) < 1e-9)
        print(f"4. 近似为0的特征值数量: {zero_count}")

        print(f"\n轴向应变 ε = {eps:.6e}")
        print(f"轴向应力 σ = {sig/1e6:.4f} MPa")
        print(f"轴力 N = {N:.2f} N")

    except Exception as e:
        print("运行异常：", e)


def verify_stiffness_column():
    """验证刚度矩阵物理意义：单自由度单位位移"""
    print("\n" + "=" * 70)
    print("========== 刚度矩阵物理意义验证 ==========")
    x1 = [0, 0, 0]
    x2 = [1, 2, 2]
    E = 210e9
    A = 2.0e-4
    _, _, Ke = truss3d_element_stiffness(x1, x2, E, A)

    # 选取第3个自由度(索引2)置1，其余为0
    de_unit = np.zeros(6)
    idx = 2
    de_unit[idx] = 1.0
    Fe = Ke @ de_unit

    print(f"设置第 {idx+1} 个自由度位移=1，其余自由度为0")
    print("节点力列阵 Fe：")
    print(np.round(Fe, 4))
    print("\n结论：")
    print("1. Fe 与刚度矩阵 Ke 的第 {} 列完全一致".format(idx+1))
    print("2. k_ij 物理含义：第j自由度单位位移、其余固定时，第i自由度所需平衡节点力")


if __name__ == "__main__":
    run_example1()
    run_example2()
    verify_stiffness_column()