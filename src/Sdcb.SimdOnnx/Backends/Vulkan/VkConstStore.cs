using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Engine;
namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Device-side const tensors (backend-packed weights) plus a 64B zero buffer, pooled per
/// (device, model, precision): every <see cref="VkGraphModel"/> of a model — any input size, any
/// output truncation — binds the same packed entries. Entries are keyed by source array and
/// consumer packing (content-compared when a plan re-materialized an array); a graph that needs
/// entries the pool lacks uploads only those, as one more chunk. Ref-counted per graph; every
/// chunk is freed when the last graph releases.
/// </summary>
internal sealed class VkConstStore
{
    private static readonly Dictionary<(VkDevice dev, object key, bool f16), VkConstStore> s_shared = new();

    private readonly VkDevice _dev;
    private readonly (VkDevice, object, bool)? _key;
    private readonly List<Entry> _entries = new();
    private readonly Dictionary<(float[] src, int kind), int> _byRef = new(RefKey.Instance);
    private readonly List<VkBuffer> _chunks = new();
    private int _refs = 1;
    internal readonly VkBuffer Dummy;
    /// <summary>Total packed bytes uploaded so far (all chunks).</summary>
    internal ulong Bytes { get; private set; }
    /// <summary>The first chunk (or <see cref="Dummy"/> while empty); identifies the pool in debug checks.</summary>
    internal VkBuffer Primary => _chunks.Count > 0 ? _chunks[0] : Dummy;

    private readonly record struct Entry(float[] Src, int Kind, int[] Shape, VkBuffer Buf, ulong Off);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private VkConstStore(VkDevice dev, (VkDevice, object, bool)? key)
    {
        _dev = dev;
        _key = key;
        Dummy = dev.NewStorageBuffer(64, hostVisible: dev.CoherentDeviceLocal);
        unsafe
        {
            byte* z = stackalloc byte[64];
            new Span<byte>(z, 64).Clear();
            dev.Upload(Dummy, z, 64);
        }
    }

    /// <summary>The pool of <paramref name="key"/> (a private one when null), with one more reference.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static VkConstStore Acquire(VkDevice dev, object? key, bool f16)
    {
        if (key is null) return new VkConstStore(dev, null);
        var k = (dev, key, f16);
        lock (s_shared)
        {
            if (s_shared.TryGetValue(k, out VkConstStore? s))
            {
                s._refs++;
                return s;
            }
            s = new VkConstStore(dev, k);
            s_shared[k] = s;
            return s;
        }
    }

    /// <summary>
    /// Buffer + byte offset of every item, uploading the missing ones as a new chunk.
    /// <paramref name="pack"/> receives the missing item indices and returns their offsets and host bytes.
    /// </summary>
    internal (VkBuffer buf, ulong off)[] Place(IReadOnlyList<(float[] src, int kind, int[] shape)> items,
        Func<int[], (ulong[] offsets, HostBytes host)> pack)
    {
        var placed = new (VkBuffer buf, ulong off)[items.Count];
        lock (_entries)
        {
            var missing = new List<int>();
            for (int i = 0; i < items.Count; i++)
            {
                int e = Find(items[i].src, items[i].kind, items[i].shape);
                if (e >= 0) placed[i] = (_entries[e].Buf, _entries[e].Off);
                else missing.Add(i);
            }
            if (missing.Count == 0) return placed;
            var (offs, host) = pack(missing.ToArray());
            VkBuffer chunk;
            using (host)
            {
                chunk = _dev.NewStorageBuffer(host.Length, hostVisible: _dev.CoherentDeviceLocal);
                unsafe { _dev.Upload(chunk, host.Ptr, host.Length); }
            }
            _chunks.Add(chunk);
            Bytes += host.Length;
            for (int j = 0; j < missing.Count; j++)
            {
                var (src, kind, shape) = items[missing[j]];
                _byRef[(src, kind)] = _entries.Count;
                _entries.Add(new Entry(src, kind, shape, chunk, offs[j]));
                placed[missing[j]] = (chunk, offs[j]);
            }
        }
        return placed;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private int Find(float[] src, int kind, int[] shape)
    {
        if (_byRef.TryGetValue((src, kind), out int e) && _entries[e].Shape.AsSpan().SequenceEqual(shape)) return e;
        // a plan may re-materialize an initializer (folding, layout): match by contents
        for (int i = 0; i < _entries.Count; i++)
        {
            Entry x = _entries[i];
            if (x.Kind == kind && x.Src.Length == src.Length && x.Shape.AsSpan().SequenceEqual(shape)
                && x.Src.AsSpan().SequenceEqual(src))
            {
                _byRef[(src, kind)] = i;
                return i;
            }
        }
        return -1;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal void Release()
    {
        lock (s_shared)
        {
            if (--_refs > 0) return;
            if (_key is { } k && s_shared.TryGetValue(k, out VkConstStore? cur) && cur == this)
                s_shared.Remove(k);
        }
        foreach (VkBuffer c in _chunks) c.Free();
        Dummy.Free();
    }

    private sealed class RefKey : IEqualityComparer<(float[] src, int kind)>
    {
        internal static readonly RefKey Instance = new();
        public bool Equals((float[] src, int kind) a, (float[] src, int kind) b) => ReferenceEquals(a.src, b.src) && a.kind == b.kind;
        [MethodImpl(MethodImplCompat.AggressiveOptimization)]
        public int GetHashCode((float[] src, int kind) k) =>
            System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(k.src) * 31 + k.kind;
    }
}
