"""PGM(P5) 读写工具，与 Go 侧 hmap 编码格式一致。所有脚本以 python3 -I 运行。"""
import numpy as np

_WS = b" \t\n\r\v\f"


def read_pgm(path):
    """读取 P5 八位灰度 PGM，返回 (size, float64[size,size] 归一化 0..1)。"""
    with open(path, "rb") as f:
        buf = f.read()

    tokens = []
    pos = 0

    def skip_ws():
        nonlocal pos
        while pos < len(buf):
            c = buf[pos]
            if c in _WS:
                pos += 1
            elif c == ord("#"):
                while pos < len(buf) and buf[pos] != ord("\n"):
                    pos += 1
            else:
                return
        raise ValueError("pgm: 头部意外结束")

    for _ in range(4):
        skip_ws()
        start = pos
        while pos < len(buf) and buf[pos] not in _WS:
            pos += 1
        tokens.append(buf[start:pos])
    magic, w, h, maxv = tokens
    if magic != b"P5" or maxv != b"255":
        raise ValueError("pgm: 仅支持 P5/8bit")
    size = int(w)
    if int(h) != size:
        raise ValueError("pgm: 仅支持方形")
    pos += 1  # maxval 后一个空白
    pix = np.frombuffer(buf[pos:pos + size * size], dtype=np.uint8).astype(np.float64)
    if pix.size != size * size:
        raise ValueError("pgm: 像素不足")
    return size, pix.reshape(size, size) / 255.0


def write_pgm(path, arr):
    """把 0..1 的二维数组量化为 P5 PGM，与 Go hmap.EncodePGM 同规则（四舍五入+截断）。"""
    a = np.asarray(arr, dtype=np.float64)
    g = np.clip(np.floor(a * 255 + 0.5), 0, 255).astype(np.uint8)
    head = f"P5\n{g.shape[1]} {g.shape[0]}\n255\n".encode()
    with open(path, "wb") as f:
        f.write(head)
        f.write(g.tobytes())
