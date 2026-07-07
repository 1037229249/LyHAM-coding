import numpy as np
from typing import List, Dict, Tuple, Optional
from Constant import ServerType, EdgeServer, SystemState


def calculate_current_energy_consumption(server: EdgeServer) -> float:
    """
    计算服务器当前时刻的总能耗 (J per time slot)
    能耗模型：
    E_total = E_base + E_cpu + E_gpu + E_memory
    Args:
        server: 边缘服务器对象
    Returns:
        float: 当前时刻总能耗 (J)
    """
    # 1. 基础功耗 (待机功耗)
    base_energy = server.base_power / 1000.0  # 转换为J (原单位mJ)

    # 2. CPU动态功耗 - 调用专用函数
    cpu_energy = calculate_cpu_energy_consumption(server)

    # 3. GPU动态功耗 - 调用专用函数 (仅AI服务器)
    gpu_energy = calculate_gpu_energy_consumption(server)

    # 4. 内存功耗 - 调用专用函数
    memory_energy = calculate_memory_energy_consumption(server)

    # 5. 实例存在额外能耗
    instance_energy = calculate_instance_existence_energy(server)

    total_energy = base_energy + cpu_energy + gpu_energy + memory_energy + instance_energy

    return total_energy


def calculate_instance_existence_energy(server: EdgeServer) -> float:
    """
    计算服务器实例存在的额外能耗
    只要服务器上有实例部署就会产生额外能耗，空闲服务器则没有
    Args:
        server: 边缘服务器对象
    Returns:
        float: 实例存在额外能耗 (J)
    """
    # 检查服务器是否有实例部署（通过可用资源计算）
    has_instances = (server.available_cpu < server.cpu_cores or
                     server.available_memory < server.memory_capacity)

    if has_instances:
        # 有实例时的额外能耗，根据服务器状态稳定估计，避免随机扰动破坏复现
        if server.server_type.value == "traditional":
            used_cpu = max(server.cpu_cores - server.available_cpu, 0)
            used_mem_ratio = (server.memory_capacity - server.available_memory) / max(server.memory_capacity, 1.0)
            return 0.05 + 0.03 * min(used_cpu / max(server.cpu_cores, 1), 1.0) + 0.02 * min(used_mem_ratio, 1.0)
        else:
            return 0.0
    else:
        return 0.0  # 空闲服务器无额外能耗

def calculate_cpu_energy_consumption(server: EdgeServer) -> float:
    """
    计算CPU动态功耗
    模型：E_cpu = P_cpu_coeff * U_cpu * C_total
    其中：
    - P_cpu_coeff: CPU功耗系数 (mJ per core per utilization)
    - U_cpu: CPU利用率 (0-1)
    - C_total: 总CPU核心数
    Args:
        server: 边缘服务器对象
    Returns:
        float: CPU动态功耗 (J)
    """
    # CPU利用率计算
    cpu_utilization = (server.cpu_cores - server.available_cpu) / max(server.cpu_cores, 1)

    # CPU动态功耗 = 功耗系数 × 利用率 × 核心数
    cpu_energy = (server.cpu_power_coeff * cpu_utilization * server.cpu_cores) / 1000.0

    return cpu_energy


def calculate_gpu_energy_consumption(server: EdgeServer) -> float:
    """
    计算GPU动态功耗（基于使用的GPU单元数）
    模型：E_gpu = P_gpu_coeff * U_gpu * GPU_units_used * GPU_active_flag
    """
    if server.server_type != ServerType.AI_CAPABLE or not server._gpu_active:
        return 0.0

    # 计算使用的GPU单元数
    used_gpu_units = server.gpu_units - server.available_gpu_units

    if used_gpu_units <= 0:
        return 0.0

    # GPU利用率计算（基于显存和存储的综合利用率）
    gpu_memory_utilization = (server.gpu_memory - server.available_gpu_memory) / max(server.gpu_memory, 1.0)
    storage_utilization = (server.model_storage - server.available_model_storage) / max(server.model_storage, 1.0)

    # 综合GPU利用率
    avg_gpu_utilization = max(gpu_memory_utilization, storage_utilization)
    avg_gpu_utilization = min(avg_gpu_utilization, 1.0)

    # GPU能耗按利用率和GPU单元数估算，作为算法无关的工程能耗口径
    if used_gpu_units == 1:
        # 单个GPU：增加基础能耗惩罚
        single_gpu_penalty = 1.20  # 增加惩罚到20%（原5%）
        gpu_energy = (server.gpu_power_coeff * avg_gpu_utilization * used_gpu_units * single_gpu_penalty) / 1000.0
    else:
        # 多个GPU：减少边际能耗节省（使多GPU优势不明显）
        import math
        if used_gpu_units >= 3:  # 3个及以上GPU时给予较少优惠
            multi_gpu_efficiency = 1.0 - 0.05 * math.log(used_gpu_units)  # 减少节省幅度（原8%）
        else:
            multi_gpu_efficiency = 1.0 - 0.03 * math.log(used_gpu_units)  # 更轻微节省（原5%）
            
        multi_gpu_efficiency = max(0.85, multi_gpu_efficiency)  # 最多降低15%（原25%）

        gpu_energy = (server.gpu_power_coeff * avg_gpu_utilization * used_gpu_units * multi_gpu_efficiency) / 1000.0

    return gpu_energy


def calculate_gpu_utilization(server: EdgeServer) -> float:
    """
    计算GPU利用率（基于使用的GPU单元数）
    """
    if server.server_type != ServerType.AI_CAPABLE or server.gpu_units == 0:
        return 0.0

    # GPU单元利用率
    gpu_unit_utilization = (server.gpu_units - server.available_gpu_units) / max(server.gpu_units, 1)

    # GPU显存利用率
    gpu_memory_utilization = (server.gpu_memory - server.available_gpu_memory) / max(server.gpu_memory, 1.0)

    # 综合GPU利用率 (取较大值，因为任一资源瓶颈都会导致高负载)
    gpu_utilization = max(gpu_unit_utilization, gpu_memory_utilization)

    return min(gpu_utilization, 1.0)


def calculate_memory_energy_consumption(server: EdgeServer) -> float:
    """
    计算内存功耗
    模型：E_memory = P_memory_base + P_memory_dynamic * U_memory
    其中：
    - P_memory_base: 内存基础功耗 (与内存容量相关)
    - P_memory_dynamic: 内存动态功耗系数
    - U_memory: 内存利用率 (0-1)
    Args:
        server: 边缘服务器对象

    Returns:
        float: 内存功耗 (J)
    """
    # 内存利用率
    memory_utilization = (server.memory_capacity - server.available_memory) / max(server.memory_capacity, 1.0)

    # 内存基础功耗 (每GB内存约0.5mJ基础功耗)
    memory_base_power = server.memory_capacity * 0.5

    # 内存动态功耗 (基于使用量的额外功耗)
    memory_dynamic_power = server.memory_capacity * memory_utilization * 0.3

    # 总内存功耗
    memory_energy = (memory_base_power + memory_dynamic_power) / 1000.0

    return memory_energy


def calculate_system_total_energy_consumption(system_state: SystemState) -> float:
    """
    计算整个系统的总能耗
    Args:
        system_state: 系统状态对象
    Returns:
        float: 系统总能耗 (J)
    """
    total_energy = 0.0

    for server in system_state.edge_servers.values():
        server_energy = calculate_current_energy_consumption(server)
        total_energy += server_energy

    return total_energy


def calculate_optimized_communication_energy(request_flow: 'RequestFlow',
                                             compression_ratio: float,
                                             server: 'EdgeServer') -> float:
    """
    计算优化压缩策略下的通信能耗
    考虑数据压缩的能耗权衡
    Args:
        request_flow: 请求流对象
        compression_ratio: 压缩比 (0.0-1.0)
        server: 边缘服务器对象
    Returns:
        float: 优化后的通信能耗 (J)
    """
    # 获取数据大小
    input_data_size = request_flow.r_input_data_size / 1000.0  # token转MB
    output_data_size = request_flow.r_output_data_size / 1000.0

    # 压缩后的数据量
    compressed_input = input_data_size * compression_ratio
    compressed_output = output_data_size * compression_ratio
    total_compressed_mb = compressed_input + compressed_output

    # 传输能耗（压缩后数据量更小）
    transmission_energy = total_compressed_mb * 0.5 / 1000.0  # 0.15mJ per MB

    # 压缩计算能耗（压缩比越高，计算开销越大）
    if compression_ratio < 1.0:
        compression_complexity = (1.0 - compression_ratio) * 3.0  # 压缩复杂度因子
        compression_energy = compression_complexity * 0.008
    else:
        compression_energy = 0.0

    total_communication_energy = transmission_energy + compression_energy

    return total_communication_energy


def calculate_cloud_processing_energy(server: 'EdgeServer', request_flow=None,
                                      f_pre: float = 1.0) -> float:
    """
    计算云端卸载时AI服务器的处理能耗
    主要是调度和协调开销

    Args:
        server: AI服务器对象
    Returns:
        float: 云端卸载处理能耗 (J)
    """
    # 云端处理不仅包含边缘侧调度，还包含远端推理租用的摊销能耗
    base_energy = (server.base_power * 0.45) / 1000.0

    # CPU调度能耗（极轻量级，约0.5%CPU用于云端协调）
    cpu_scheduling_energy = (server.cpu_power_coeff * 0.005) / 1000.0

    # 网络接口能耗
    network_interface_energy = 0.001  # 固定1mJ用于网络协调

    remote_inference_energy = 0.0
    preprocess_energy = 0.0
    if request_flow is not None:
        token_load = max(
            float(getattr(request_flow, "r_input_data_size", 128.0)) +
            float(getattr(request_flow, "r_output_data_size", 32.0)),
            1.0
        )
        arrival_rate = max(float(getattr(request_flow, "arrival_rate", 1.0)), 1.0)
        # 远端推理按token和请求强度摊销。云端侧采用更高能效的共享AI加速器，
        # 因此单位token能耗低于边缘本地小GPU；通信能耗仍单独计入。
        remote_inference_energy = 0.00048 * token_load * min(arrival_rate / 8.0, 2.5)
        preprocess_energy = 0.006 * float(f_pre) ** 2 * min(token_load / 512.0, 3.0)

    total_processing_energy = (
        base_energy + cpu_scheduling_energy + network_interface_energy +
        remote_inference_energy + preprocess_energy
    )

    return total_processing_energy


def calculate_local_ai_processing_energy(server: 'EdgeServer',
                                         gpu_units: int,
                                         gpu_memory: float,
                                         model_storage: float) -> float:
    """
    计算本地AI处理的能耗
    【修复】正确应用单GPU和多GPU的能耗策略
    """
    # 基础功耗
    base_energy = server.base_power / 1000.0

    # CPU调度能耗（本地AI推理需要CPU调度）
    cpu_scheduling_energy = (server.cpu_power_coeff * 0.05) / 1000.0

    # 【修复】应用正确的GPU能耗策略
    if gpu_units > 0:
        # 1. 计算GPU利用率
        gpu_utilization = min(gpu_memory / server.gpu_memory, 1.0)

        # 2. 临时设置服务器状态以使用统一的GPU能耗计算
        original_available_gpu_units = server.available_gpu_units
        original_available_gpu_memory = server.available_gpu_memory
        original_gpu_active = server._gpu_active

        # 模拟GPU资源使用状态
        server.available_gpu_units = server.gpu_units - gpu_units
        server.available_gpu_memory = server.gpu_memory - gpu_memory
        server._gpu_active = True

        # 3. 调用统一的GPU能耗计算函数（包含单GPU/多GPU策略）
        gpu_energy = calculate_gpu_energy_consumption(server)

        # 4. 恢复原始状态
        server.available_gpu_units = original_available_gpu_units
        server.available_gpu_memory = original_available_gpu_memory
        server._gpu_active = original_gpu_active

    else:
        gpu_energy = 0.0

    # 内存访问能耗（模型存储访问）
    storage_utilization = min(model_storage / server.model_storage, 1.0)
    memory_access_energy = storage_utilization * 0.05

    total_local_energy = base_energy + cpu_scheduling_energy + gpu_energy + memory_access_energy

    return total_local_energy


def calculate_comprehensive_ai_energy(server: 'EdgeServer',
                                      processing_mode: str,
                                      request_flow: 'RequestFlow',
                                      **kwargs) -> float:
    """
    计算AI微服务的综合能耗
    新增函数：根据处理模式调用不同的能耗计算函数

    Args:
        server: AI服务器对象
        processing_mode: 处理模式 ('local_processing' 或 'cloud_offloaded')
        request_flow: 请求流对象
        **kwargs: 其他参数
    Returns:
        float: AI微服务综合能耗 (J)
    """
    if processing_mode == 'local_processing':
        # 本地处理能耗
        gpu_units = kwargs.get('gpu_units', 1)
        gpu_memory = kwargs.get('gpu_memory', 16.0)
        model_storage = kwargs.get('model_storage', 20.0)

        return calculate_local_ai_processing_energy(server, gpu_units, gpu_memory, model_storage)

    elif processing_mode == 'cloud_offloaded':
        # 云端卸载能耗
        compression_ratio = kwargs.get('compression_ratio', 1.0)

        # 通信能耗
        communication_energy = calculate_optimized_communication_energy(
            request_flow, compression_ratio, server)

        # 云端处理协调能耗
        processing_energy = calculate_cloud_processing_energy(server)

        return communication_energy + processing_energy

    else:
        # 未知模式，返回基础能耗
        return server.base_power / 1000.0
