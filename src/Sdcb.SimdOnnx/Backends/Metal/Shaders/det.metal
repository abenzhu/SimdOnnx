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
// flags: bit1 add same-shape a, bit2 relu on a (FPN top-down merge: o = up(x) + [relu](a)).
struct PcNearest { uint c, ih, iw, oh, ow, flags; float sy, sx; };
kernel void nearest(device const half* x [[buffer(0)]],
                    device const half* a [[buffer(1)]],
                    device half* o [[buffer(2)]],
                    constant PcNearest& p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.oh * p.ow * p.c) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    uint iy = min(uint(floor(float(oy) / p.sy)), p.ih - 1u);
    uint ix = min(uint(floor(float(ox) / p.sx)), p.iw - 1u);
    float v = float(x[(iy * p.iw + ix) * p.c + ch]);
    if ((p.flags & 2u) != 0u)
    {
        float s = float(a[gid]);
        v += (p.flags & 4u) != 0u ? max(s, 0.0f) : s;
    }
    o[gid] = half(v);
}
kernel void nearest4(device const half4* x [[buffer(0)]],
                     device const half4* a [[buffer(1)]],
                     device half4* o [[buffer(2)]],
                     constant PcNearest& p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= p.oh * p.ow * c4) return;
    uint q = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    uint iy = min(uint(floor(float(oy) / p.sy)), p.ih - 1u);
    uint ix = min(uint(floor(float(ox) / p.sx)), p.iw - 1u);
    float4 v = float4(x[(iy * p.iw + ix) * c4 + q]);
    if ((p.flags & 2u) != 0u)
    {
        float4 s = float4(a[gid]);
        v += (p.flags & 4u) != 0u ? max(s, float4(0.0f)) : s;
    }
    o[gid] = half4(v);
}

// ---- se_gate: squeeze-excitation gate, one threadgroup per image ----
// mean over the image's pixels -> h = act1(W1 mean + b1) -> s = gate(W2 h + b2) + offset, [img * c + ch].
// W1 [r][c], W2 [c][r], biases fp32. flags: bit0 b1, bit1 b2, bit2 relu on h, bit3 hard gate (else sigmoid).
struct PcSeGate { uint c, r, hw, flags; float inv, alpha, beta, offset; };
kernel void se_gate(device const half* x [[buffer(0)]],
                    device const float* w1 [[buffer(1)]],
                    device const float* b1 [[buffer(2)]],
                    device const float* w2 [[buffer(3)]],
                    device const float* b2 [[buffer(4)]],
                    device half* o [[buffer(5)]],
                    constant PcSeGate& p [[buffer(6)]],
                    uint img [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float mean[2048];
    threadgroup float hid[1024];
    threadgroup float red[256];
    uint lx = tid & 63u, ly = tid >> 6;
    device const half* xi = x + (ulong)img * p.hw * p.c;
    for (uint c0 = 0u; c0 < p.c; c0 += 64u)
    {
        uint ch = c0 + lx;
        float s = 0.0f;
        if (ch < p.c)
            for (uint px = ly; px < p.hw; px += 4u) s += float(xi[(ulong)px * p.c + ch]);
        red[tid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ly == 0u && ch < p.c) mean[ch] = (red[lx] + red[lx + 64u] + red[lx + 128u] + red[lx + 192u]) * p.inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = tid; j < p.r; j += 256u)
    {
        float s = (p.flags & 1u) != 0u ? b1[j] : 0.0f;
        for (uint k = 0u; k < p.c; k++) s += w1[j * p.c + k] * mean[k];
        hid[j] = (p.flags & 4u) != 0u ? max(s, 0.0f) : s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ch = tid; ch < p.c; ch += 256u)
    {
        float z = (p.flags & 2u) != 0u ? b2[ch] : 0.0f;
        for (uint j = 0u; j < p.r; j++) z += w2[ch * p.r + j] * hid[j];
        float g = (p.flags & 8u) != 0u ? clamp(p.alpha * z + p.beta, 0.0f, 1.0f) : 1.0f / (1.0f + exp(-z));
        o[img * p.c + ch] = half(g + p.offset);
    }
}
