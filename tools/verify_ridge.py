#!/usr/bin/env python3
# 独立复核：不引用 Go 代码，直接读 height_*.f64（小端 float64），
# 重新实现曲率、符号翻转判定、差分相关预判与连通段归类，
# 并与 Go 输出的掩码 PGM 逐像素比对。
#
# 用法: python3 -I tools/verify_ridge.py <产物目录> <画幅>
# 注意必须用浮点高度：8 位 PGM 的量化步长大于曲率均值，会把判定毁掉。
import sys, struct, math

def parse_pgm_mask(p):
    b = open(p, 'rb').read()
    toks, i = [], 0
    while len(toks) < 4:
        while b[i:i+1].isspace(): i += 1
        s = i
        while not b[i:i+1].isspace(): i += 1
        toks.append(b[s:i])
    i += 1
    w, h, mx = int(toks[1]), int(toks[2]), int(toks[3])
    return w, h, mx, b[i:i+w*h]

def main(n):
    root = sys.argv[1]
    raw = open(f'{root}/height_{n}.f64', 'rb').read()
    assert len(raw) == n*n*8, (len(raw), n*n*8)
    hgt = list(struct.unpack(f'<{n*n}d', raw))

    kap = [[None]*n for _ in range(n)]
    for y in range(n):
        for x in range(1, n-1):
            kap[y][x] = hgt[y*n+x-1] - 2*hgt[y*n+x] + hgt[y*n+x+1]

    mask = [[False]*n for _ in range(n)]
    truth = 0
    for y in range(n):
        for x in range(2, n-2):
            dl = kap[y][x] - kap[y][x-1]
            dr = kap[y][x+1] - kap[y][x]
            if dl < 0 and dr > 0:
                mask[y][x] = True
                truth += 1

    sx = []; sy = []
    for y in range(n):
        for x in range(1, n-3):
            sx.append(kap[y][x+1] - kap[y][x])
            sy.append(kap[y][x+2] - kap[y][x+1])
    m = len(sx)
    mx = sum(sx)/m; my = sum(sy)/m
    vx = sum((a-mx)**2 for a in sx)/m
    vy = sum((b-my)**2 for b in sy)/m
    cxy = sum((a-mx)*(b-my) for a, b in zip(sx, sy))/m
    rho = cxy/math.sqrt(vx*vy)
    p = 0.25 - math.asin(rho)/(2*math.pi)
    pred = p*n*(n-4)

    w, h, maxv, body = parse_pgm_mask(f'{root}/ridge_mask_{n}.pgm')
    assert (w, h) == (n, n)
    mism = 0
    white = 0
    for y in range(n):
        for x in range(n):
            on = body[y*n+x] > 0
            if on: white += 1
            if on != mask[y][x]: mism += 1

    # 8 连通段
    seen = [[False]*n for _ in range(n)]
    sizes = []
    for y in range(n):
        for x in range(n):
            if not mask[y][x] or seen[y][x]: continue
            stack = [(x, y)]; seen[y][x] = True; c = 0
            while stack:
                cx, cy = stack.pop(); c += 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < n and 0 <= ny < n and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True; stack.append((nx, ny))
            sizes.append(c)
    sizes.sort(reverse=True)

    print(f"N={n} truth={truth} white={white} mismatch={mism} rho={rho:+.5f} pred={pred:.1f} dev={100*(pred-truth)/truth:+.2f}%")
    print(f"      components={len(sizes)} top5={sizes[:5]}")

main(int(sys.argv[2]))
