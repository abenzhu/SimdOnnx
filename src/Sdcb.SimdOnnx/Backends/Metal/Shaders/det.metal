// PP-OCR det ops (HardSigmoid, global average pool, stride==kernel transposed conv,
// nearest resize). fp16 storage, fp32 math — see common.metal.

// ---- hsig: o = clamp(alpha*x + beta, 0, 1) ----
struct PcHsig { uint n; float alpha, beta; };
kernel void hsig(device const half* x [[buffer(0)]],
                 device half* o [[buffer(1)]],
                 constant PcHsig& p [[buffer(2)]],
                 uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    o[gid] = half(clamp(float(x[gid]) * p.alpha + p.beta, 0.0f, 1.0f));
}

// ---- gap / gap4: NHWC global average pool, o[ch] = mean over hw of x[px*c + ch] ----
// gap: 64 channels x 4 pixel lanes per threadgroup; gap4 (c % 4 == 0): 16 quads x 16 pixel lanes.
struct PcGap { uint c, hw; float inv; };
kernel void gap(device const half* x [[buffer(0)]],
                device half* o [[buffer(1)]],
                constant PcGap& p [[buffer(2)]],
                uint wg [[threadgroup_position_in_grid]],
                uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float part[256];
    uint lx = tid & 63u, ly = tid >> 6;
    uint ch = wg * 64u + lx;
    float s = 0.0f;
    if (ch < p.c)
        for (uint px = ly; px < p.hw; px += 4u) s += float(x[px * p.c + ch]);
    part[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ly == 0u && ch < p.c)
        o[ch] = half((part[lx] + part[lx + 64u] + part[lx + 128u] + part[lx + 192u]) * p.inv);
}
kernel void gap4(device const half4* x [[buffer(0)]],
                 device half4* o [[buffer(1)]],
                 constant PcGap& p [[buffer(2)]],
                 uint wg [[threadgroup_position_in_grid]],
                 uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float4 part[256];
    uint c4 = p.c >> 2;
    uint lx = tid & 15u, ly = tid >> 4;
    uint q = wg * 16u + lx;
    float4 s = float4(0.0f);
    if (q < c4)
        for (uint px = ly; px < p.hw; px += 16u) s += float4(x[px * c4 + q]);
    part[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ly == 0u && q < c4)
    {
        float4 t = float4(0.0f);
        for (uint k = 0u; k < 16u; k++) t += part[k * 16u + lx];
        o[q] = half4(t * p.inv);
    }
}

// ---- convt: transposed conv, stride == kernel, no pad (non-overlapping taps) ----
// o[oy,ox,oc] = act(b[oc] + sum_c x[oy/kh, ox/kw, c] * w[(tap*ic + c)*oc + oc]);
// flags: 1 bias, (flags >> 1) act: 1 relu, 2 sigmoid.
struct PcConvT { uint ic, ih, iw, oc, kh, kw, flags; };
kernel void convt(device const half* x [[buffer(0)]],
                  device const half* w [[buffer(1)]],
                  device const float* b [[buffer(2)]],
                  device half* o [[buffer(3)]],
                  constant PcConvT& p [[buffer(4)]],
                  uint gid [[thread_position_in_grid]])
{
    uint ow = p.iw * p.kw, oh = p.ih * p.kh;
    if (gid >= oh * ow * p.oc) return;
    uint oc = gid % p.oc, px = gid / p.oc;
    uint ox = px % ow, oy = px / ow;
    uint tap = (oy % p.kh) * p.kw + ox % p.kw;
    uint xb = ((oy / p.kh) * p.iw + ox / p.kw) * p.ic, wb = tap * p.ic * p.oc + oc;
    float s = (p.flags & 1u) != 0u ? b[oc] : 0.0f;
    for (uint c = 0u; c < p.ic; c++) s += float(x[xb + c]) * float(w[wb + c * p.oc]);
    uint act = p.flags >> 1;
    if (act == 1u) s = max(s, 0.0f);
    else if (act == 2u) s = 1.0f / (1.0f + fexp1(-s));
    o[gid] = half(s);
}

// ---- nearest / nearest4: NHWC nearest resize, asymmetric, floor: src = floor(dst / scale) ----
struct PcNearest { uint c, ih, iw, oh, ow; float sy, sx; };
kernel void nearest(device const half* x [[buffer(0)]],
                    device half* o [[buffer(1)]],
                    constant PcNearest& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.oh * p.ow * p.c) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    uint iy = min(uint(floor(float(oy) / p.sy)), p.ih - 1u);
    uint ix = min(uint(floor(float(ox) / p.sx)), p.iw - 1u);
    o[gid] = x[(iy * p.iw + ix) * p.c + ch];
}
kernel void nearest4(device const half4* x [[buffer(0)]],
                     device half4* o [[buffer(1)]],
                     constant PcNearest& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= p.oh * p.ow * c4) return;
    uint q = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    uint iy = min(uint(floor(float(oy) / p.sy)), p.ih - 1u);
    uint ix = min(uint(floor(float(ox) / p.sx)), p.iw - 1u);
    o[gid] = x[(iy * p.iw + ix) * c4 + q];
}
