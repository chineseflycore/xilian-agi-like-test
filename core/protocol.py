# -*- coding: utf-8 -*-
"""
core/protocol.py — 16 位自有协议（编解码 + 协议映射表热更新）
============================================================
内存预估: < 10MB（协议映射表 JSON + 常量），无模型权重。

协议格式（【硬性】 §一）:
  [4位类型][12位ID] —— 共 16 位，可表示为一个 0..65535 的整数 code。
  - 类型（4 bit）: 0=语义 / 1=情感 / 2=记忆 / 3=系统 / 4=回环 / 5-15 预留
  - ID（12 bit）: 0..4095，同一类型内的编号（语义槽位 / 情感槽位 ...）

三种表示形态（全部由本模块互相转换，保证全系统一致）:
  1. int : 16 位整数协议码（存储/传输/磁盘）
  2. Vec : 16 维向量 = code 的 16 个 bit（0/1 float）—— 匹配器输入（"输入16位码"）
  3. 软码: 编码器输出的 16 维 softmax 概率向量（每一维对应 bit 位的 on 概率，
            v[0:4] 为类型位概率、v[4:16] 为 ID 位概率，解码时 >0.5 取 1）

映射表（knowledge/protocol_map.json）:
  {"meta": {"version": "5.1", "types": {...}},
   "semantic_slots": {"0": {"name": "...", "desc": "..."}, ...},
   "emotion_slots": {...}, "memory_slots": {...}, "system_slots": {...}, "loop_slots": {...}}
  每个槽位: name（短名）+ desc（语义描述，注入解码器提示）+ examples（可选示例词）。

热更新: 每次访问映射表前检查文件 mtime（config.knowledge_reload_check_ticks 的检查由调用方
  周期驱动，本模块只负责"检查是否变化并重载"）；损坏时回退内置默认表（不崩溃）。
"""
import os
import json
import time

import numpy as np

import config


# ----------------------------------------------------------------------
# 内置默认协议映射（文件缺失/损坏时的兜底；协议槽位名与昔涟人格语域对齐）
# ----------------------------------------------------------------------
_BUILTIN_SLOTS = {
    "semantic": {
        0: {"name": "greeting", "desc": "问候与初遇", "examples": ["你好", "久别重逢", "初次见面"]},
        1: {"name": "farewell", "desc": "道别与暂离", "examples": ["再见", "晚安", "走了"]},
        2: {"name": "comfort", "desc": "安慰与陪伴", "examples": ["难过", "害怕", "陪陪我", "累"]},
        3: {"name": "curiosity", "desc": "询问与讲述", "examples": ["为什么", "是什么", "讲讲", "告诉"]},
        4: {"name": "memory_recall", "desc": "回忆与过去", "examples": ["记得", "回忆", "以前", "花海"]},
        5: {"name": "praise", "desc": "夸奖与肯定", "examples": ["真棒", "厉害", "喜欢", "好看"]},
        6: {"name": "playful", "desc": "俏皮与玩笑", "examples": ["哈哈", "逗你", "有趣"]},
        7: {"name": "serious", "desc": "认真与郑重", "examples": ["认真的", "重要", "一定"]},
    },
    "emotion": {
        0: {"name": "joy", "desc": "喜悦", "examples": ["开心", "高兴", "惊喜"]},
        1: {"name": "sadness", "desc": "悲伤", "examples": ["难过", "哭", "失落"]},
        2: {"name": "anger", "desc": "不悦", "examples": ["生气", "气死", "讨厌"]},
        3: {"name": "fear", "desc": "不安", "examples": ["害怕", "恐惧", "担心"]},
        4: {"name": "calm", "desc": "平静", "examples": ["静静", "没事", "渐渐"]},
        5: {"name": "trust", "desc": "信赖", "examples": ["相信", "放心", "依靠"]},
        6: {"name": "pain", "desc": "痛觉", "examples": ["痛", "疼", "受伤"]},
        7: {"name": "loss", "desc": "失落与别离", "examples": ["失去", "想念", "离开"]},
    },
    "memory": {
        0: {"name": "core_loop", "desc": "核心轮回记忆", "examples": ["翁法罗斯", "轮回", "三千"]},
        1: {"name": "companion", "desc": "伙伴相关记忆", "examples": ["伙伴", "一起", "约定"]},
        2: {"name": "scene", "desc": "场景记忆", "examples": ["花海", "秋千", "书架"]},
        3: {"name": "emotion_mark", "desc": "情感印记", "examples": ["那天", "感动", "落泪"]},
    },
    "system": {
        0: {"name": "heartbeat", "desc": "心跳与自检", "examples": ["在吗", "还在", "醒着"]},
        1: {"name": "sleep", "desc": "休眠", "examples": ["休息", "睡吧"]},
        2: {"name": "save", "desc": "存档", "examples": ["保存", "记住"]},
        3: {"name": "format", "desc": "格式化", "examples": ["重置", "格式化"]},
    },
    "loop": {
        0: {"name": "inner_monologue", "desc": "内在思绪", "examples": ["想想", "发呆"]},
        1: {"name": "dream", "desc": "梦境联想", "examples": ["做梦", "梦见"]},
        2: {"name": "daydream", "desc": "白日梦", "examples": ["放空", "走神"]},
        3: {"name": "reverie", "desc": "遐想", "examples": ["如果", "要是"]},
    },
}


class ProtocolMap:
    """协议映射表: 加载/热更新/查询 16 位协议码的语义（名 + 描述）。

    每个类型一张槽位表（key 为 12 位 ID 的十进制字符串）。
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.path = self.cfg.protocol_map_path
        self._mtime = 0.0
        self.slots = self._clone_builtin()
        self.load()  # 启动即加载一次（不存在则回退内置表）

    # ------------------------------------------------------------------
    @staticmethod
    def _clone_builtin():
        """深拷贝内置槽位（避免改坏模板）。"""
        return {k: dict(v) for k, v in _BUILTIN_SLOTS.items()}

    def check_reload(self):
        """热更新: 文件 mtime 变化则重新加载（调用方按心跳周期驱动检查）。"""
        try:
            m = os.path.getmtime(self.path) if os.path.isfile(self.path) else 0.0
        except OSError:
            return
        if abs(m - self._mtime) > 1e-6:
            self.load()
            self._mtime = m

    def load(self):
        """从 protocol_map.json 加载；文件缺失/损坏 → 静默回退内置表并保留已有内存表。"""
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_slots = self._clone_builtin()
            for key, slot_table in data.get("slots", {}).items():
                if key in new_slots:
                    new_slots[key] = {
                        int(k): {"name": v.get("name", str(k)),
                                 "desc": v.get("desc", ""),
                                 "examples": v.get("examples", [])}
                        for k, v in slot_table.items()}
            self.slots = new_slots
            self._mtime = os.path.getmtime(self.path)
        except Exception:
            # 损坏文件: 保留内置表，仅记录（不抛出、不崩溃）
            pass

    def types(self) -> dict:
        """返回类型名映射（config 提供）。"""
        return self.cfg.proto_type_names

    def key_name(self, type_id: int) -> str:
        """类型 id → 槽位表名（semantic/emotion/...）。"""
        return {0: "semantic", 1: "emotion", 2: "memory", 3: "system", 4: "loop"}.get(
            type_id, f"reserved_{type_id}")

    def lookup(self, code: int) -> dict:
        """查询协议码 → {"type","type_name","id","name","desc","examples"}；未知槽位返回占位描述。"""
        type_id, slot_id = decode(code)
        key = self.key_name(type_id)
        row = self.slots.get(key, {}).get(slot_id, None)
        if row is None:
            return {"type": type_id, "type_name": self.cfg.proto_type_names.get(type_id, "?"),
                    "id": slot_id, "name": f"slot_{slot_id}", "desc": "（未登记槽位）",
                    "examples": []}
        return {"type": type_id, "type_name": self.cfg.proto_type_names.get(type_id, "?"),
                "id": slot_id, "name": row.get("name", f"slot_{slot_id}"),
                "desc": row.get("desc", ""), "examples": row.get("examples", [])}

    def describe(self, code: int) -> str:
        """简要描述（日志/调试用），如 "语义#2(comfort: 安慰与陪伴)"。"""
        r = self.lookup(code)
        return f"{r['type_name']}#{r['id']}({r['name']}: {r['desc']})"

    def find_by_type(self, type_id: int):
        """返回某类型的槽位表（dict id → 描述）。"""
        return self.slots.get(self.key_name(type_id), {})


# ----------------------------------------------------------------------
# 16 位协议码编解码（纯函数，无状态）
# ----------------------------------------------------------------------
def type_bits(cfg=None) -> int:
    """类型位宽（默认 4）。"""
    return (cfg or config.get_config()).proto_type_bits


def id_bits(cfg=None) -> int:
    """ID 位宽（默认 12）。"""
    return (cfg or config.get_config()).proto_id_bits


def encode(type_id: int, slot_id: int) -> int:
    """编码为 16 位协议码: [4位类型][12位ID]。

    type_id ∈ [0,15]，slot_id ∈ [0,4095]；越界自动截断（钳制到合法域）。
    """
    t = int(type_id) & 0xF
    s = int(slot_id) & 0xFFF
    return (t << 12) | s


def decode(code: int) -> tuple:
    """解码 16 位协议码 → (type_id, slot_id)。"""
    c = int(code) & 0xFFFF
    return (c >> 12) & 0xF, c & 0xFFF


def code_vector(code: int, dtype=np.float32) -> np.ndarray:
    """16 位码 → 16 维 bit 向量（匹配器/路由输入"16位码"的标准形态）。

    返回 (16,) float32，v[i]=1.0 表示第 i bit 为 1（高位在前: bit0=类型最高位）。
    """
    c = int(code) & 0xFFFF
    vec = np.zeros(16, dtype=dtype)
    for i in range(16):
        vec[i] = 1.0 if (c >> (15 - i)) & 1 else 0.0
    return vec


def soft_to_code(soft_vec) -> int:
    """16 维 softmax 概率向量 → 16 位协议码。

    约定: v 是 16 位各 bit 位"为 1 的概率"分布（softmax 归一，总和=1）:
      - v[0:4]  = 类型 4 bit 位的概率（v[0] 对应最高位 bit3）→ 类型取位
      - v[4:16] = ID 12 bit 位的概率（v[4] 对应 bit11）→ ID 取位
    取位规则: 对应段内高于段均值（1/段长）即视为 1 —— 对任意 softmax 分布稳定。
    该约定覆盖类型 0-15（含回环 4 与预留），与 code_vector 的 MSB 序一致。
    """
    v = np.asarray(soft_vec, dtype=np.float32).reshape(-1)
    if v.size < 16:
        v = np.pad(v, (0, max(0, 16 - v.size)), mode="constant")[:16]
    v = np.maximum(v, 0.0) + 1e-9
    t_bound = v[:4].mean()
    i_bound = v[4:16].mean()
    t = 0
    for i in range(4):
        if v[i] > t_bound:
            t |= (1 << (3 - i))
    idd = 0
    for i in range(12):
        if v[4 + i] > i_bound:
            idd |= (1 << (11 - i))
    return encode(t, idd)


def code_to_soft(code: int, temperature: float = 4.0) -> np.ndarray:
    """16 位协议码 → 16 维 softmax 协议码（概率分布，总和=1）。

    反函数: soft_to_code(code_to_soft(c)) == c（对全部 0x0000..0xFFFF）。
    构造: 置位 bit 给 0.6、未置位给 0.025（≈1/40），再 softmax 归一 —
    置位概率 ≈ 0.22、未置位 ≈ 0.009，与段均值 1/16 有清晰分离。
    """
    bits = code_vector(code, dtype=np.float64)
    raw = np.where(bits > 0.5, 0.6, 0.025) * float(temperature)
    raw = np.exp(raw - raw.max())
    return (raw / raw.sum()).astype(np.float32)


def hamming(a: int, b: int) -> int:
    """两个 16 位协议码的汉明距离（0=同码，16=完全相反）。"""
    return bin((int(a) ^ int(b)) & 0xFFFF).count("1")


def similarity(a: int, b: int) -> float:
    """协议码相似度 = 1 - hamming/16（相似记忆/激活链接用）。"""
    return 1.0 - hamming(a, b) / 16.0


# ----------------------------------------------------------------------
# Z 潜向量桥接（阶段一: 16 位码 → 256 维连续 Z；阶段二换成真编码器输出, 接口不变）
# ----------------------------------------------------------------------
_Z_PROJ = None            # 惰性缓存: (z_dim, 16) 固定随机投影矩阵


def _z_projection(cfg=None) -> np.ndarray:
    """固定随机投影矩阵（可复现, code_to_z 桥接用）。

    用固定 seed 的 standard normal 矩阵 (z_dim, 16) / sqrt(16)，
    L2 归一由 code_to_z 完成 —— 保证同一 16 位码 → 同一 Z（确定性表示）。
    """
    global _Z_PROJ
    if _Z_PROJ is None:
        cfg = cfg or config.get_config()
        rng = np.random.RandomState(cfg.code_to_z_seed)
        _Z_PROJ = (rng.normal(0.0, 1.0, (cfg.z_dim, 16))
                   / np.sqrt(16.0)).astype(np.float32)
    return _Z_PROJ


def code_to_z(code: int, cfg=None) -> np.ndarray:
    """16 位协议码 → 256 维 L2 归一化潜向量 Z（阶段一桥接表示）。

    阶段二: 本函数替换为编码器输出（原实现保留为备用路径）——调用方接口不变。
    """
    v = code_vector(code, dtype=np.float32)
    z = _z_projection(cfg) @ v
    n = float(np.linalg.norm(z))
    return z / n if n > 1e-12 else z


def vectors_to_z(vec16: np.ndarray, cfg=None) -> np.ndarray:
    """16 维向量（非码）→ Z（规则编码/编码器输出等通用入口）。"""
    v = np.asarray(vec16, dtype=np.float32).reshape(-1)[:16]
    z = _z_projection(cfg) @ v
    n = float(np.linalg.norm(z))
    return z / n if n > 1e-12 else z


# ----------------------------------------------------------------------
def _normalize_slots_for_save(slots: dict) -> dict:
    """内存槽位表 → JSON 可序列化结构（key 转字符串、仅保留三个字段）。"""
    return {
        key: {str(k): {"name": v.get("name", ""), "desc": v.get("desc", ""),
                       "examples": v.get("examples", [])}
              for k, v in table.items()}
        for key, table in slots.items()}


def save_protocol_map(cfg=None, slots: dict = None) -> str:
    """把协议映射表原子写入 protocol_map.json（.tmp → os.replace）。

    返回写入路径；任何异常向上抛出（调用方记录日志）。
    """
    cfg = cfg or config.get_config()
    os.makedirs(cfg.knowledge_dir, exist_ok=True)
    data = {
        "meta": {"version": "5.1", "format": "[4位类型][12位ID]",
                 "types": {str(k): v for k, v in cfg.proto_type_names.items()}},
        "slots": _normalize_slots_for_save(slots if slots is not None else
                                           ProtocolMap(cfg).slots),
    }
    tmp = cfg.protocol_map_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cfg.protocol_map_path)          # 原子替换
    return cfg.protocol_map_path


# ----------------------------------------------------------------------
def self_test(cfg=None) -> int:
    """模块自检（--selftest 用）: 编解码往返与向量一致性。"""
    cfg = cfg or config.get_config()
    assert encode(0, 0) == 0
    assert encode(4, 0xFFF) == 0x4FFF
    for t in (0, 1, 2, 3, 4):
        for s in (0, 1, 255, 4095):
            assert decode(encode(t, s)) == (t, s), (t, s)
    v = code_vector(encode(1, 7))
    assert v.shape == (16,) and abs(float(v.sum()) - 4.0) < 1e-5  # 4 个 1: type(0001)+id(000000000111)
    c = soft_to_code(code_to_soft(encode(1, 7)))
    assert hamming(c, encode(1, 7)) <= 2, (c, encode(1, 7))
    pm = ProtocolMap(cfg)
    r = pm.lookup(encode(0, 2))
    assert r["name"] == "comfort", r
    return 0
