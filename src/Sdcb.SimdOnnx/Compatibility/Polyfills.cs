// netstandard2.0 only (see the csproj): BCL members the engine uses that netstandard2.0 lacks,
// supplied as C# 14 extension members so the shared sources call them unchanged.
#pragma warning disable CS1591
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sdcb.SimdOnnx
{
    internal static class Polyfills
    {
        extension(Math)
        {
            public static int Clamp(int v, int lo, int hi) => v < lo ? lo : v > hi ? hi : v;
            public static long Clamp(long v, long lo, long hi) => v < lo ? lo : v > hi ? hi : v;
            public static float Clamp(float v, float lo, float hi) => v < lo ? lo : v > hi ? hi : v;
            public static double Clamp(double v, double lo, double hi) => v < lo ? lo : v > hi ? hi : v;
        }

        extension(Array)
        {
            public static void Fill<T>(T[] a, T value)
            {
                for (int i = 0; i < a.Length; i++) a[i] = value;
            }

            public static void Fill<T>(T[] a, T value, int start, int count)
            {
                for (int i = start; i < start + count; i++) a[i] = value;
            }

            public static void Clear(Array a) => Array.Clear(a, 0, a.Length);
        }

        extension(BitConverter)
        {
            public static float Int32BitsToSingle(int v) => Unsafe.As<int, float>(ref v);
            public static int SingleToInt32Bits(float v) => Unsafe.As<float, int>(ref v);
        }

        extension(float)
        {
            public static bool IsFinite(float f) => !float.IsNaN(f) && !float.IsInfinity(f);
        }

        extension(string)
        {
            public static string Join<T>(char separator, IEnumerable<T> values) => string.Join(separator.ToString(), values);
            public static string Join(char separator, params string[] values) => string.Join(separator.ToString(), values);
        }

        extension<TKey, TValue>(IDictionary<TKey, TValue> d)
        {
            public TValue? GetValueOrDefault(TKey key) => d.TryGetValue(key, out TValue? v) ? v : default;
            public TValue GetValueOrDefault(TKey key, TValue fallback) => d.TryGetValue(key, out TValue? v) ? v : fallback;
        }

        extension(GC)
        {
            /// <summary>Pinned arrays stay pinned for the process lifetime (the handle is never freed);
            /// only grow-only thread-local scratch slots use this.</summary>
            public static T[] AllocateUninitializedArray<T>(int length, bool pinned = false)
            {
                var a = new T[length];
                if (pinned) GCHandle.Alloc(a, GCHandleType.Pinned);
                return a;
            }
        }

        extension(MemoryMarshal)
        {
            public static ref T GetArrayDataReference<T>(T[] array) => ref MemoryMarshal.GetReference(array.AsSpan());
        }

        extension(Interlocked)
        {
            public static nint Exchange(ref nint location, nint value) => IntPtr.Size == 8
                ? (nint)Interlocked.Exchange(ref Unsafe.As<nint, long>(ref location), (long)value)
                : (nint)Interlocked.Exchange(ref Unsafe.As<nint, int>(ref location), (int)value);
        }

        extension(Vector)
        {
            public static Vector<T> LoadUnsafe<T>(ref T source) where T : struct =>
                Unsafe.ReadUnaligned<Vector<T>>(ref Unsafe.As<T, byte>(ref source));

            public static T Sum<T>(Vector<T> v) where T : struct => Vector.Dot(v, Vector<T>.One);
        }

        extension<T>(Vector<T> v) where T : struct
        {
            public void StoreUnsafe(ref T destination) =>
                Unsafe.WriteUnaligned(ref Unsafe.As<T, byte>(ref destination), v);
        }

        extension(Stream s)
        {
            public int Read(Span<byte> buffer)
            {
                byte[] tmp = new byte[buffer.Length];
                int n = s.Read(tmp, 0, tmp.Length);
                tmp.AsSpan(0, n).CopyTo(buffer);
                return n;
            }

            public void ReadExactly(Span<byte> buffer)
            {
                byte[] tmp = new byte[buffer.Length];
                int got = 0;
                while (got < tmp.Length)
                {
                    int n = s.Read(tmp, got, tmp.Length - got);
                    if (n <= 0) throw new EndOfStreamException();
                    got += n;
                }
                tmp.CopyTo(buffer);
            }
        }
    }
}

namespace System.Runtime.InteropServices
{
    /// <summary>netstandard2.0 stand-in over the process heap (Marshal.AllocHGlobal).</summary>
    internal static unsafe class NativeMemory
    {
        public static void* Alloc(nuint bytes) => (void*)Marshal.AllocHGlobal((nint)Math.Max(1UL, (ulong)bytes));

        public static void* AllocZeroed(nuint bytes)
        {
            void* p = Alloc(bytes);
            for (ulong off = 0; off < bytes; off += uint.MaxValue)
                Unsafe.InitBlockUnaligned((byte*)p + off, 0, (uint)Math.Min(uint.MaxValue, bytes - off));
            return p;
        }

        public static void Free(void* p)
        {
            if (p != null) Marshal.FreeHGlobal((nint)p);
        }

        public static void* AlignedAlloc(nuint bytes, nuint alignment)
        {
            // over-allocate, keep the raw pointer just below the aligned block
            nint raw = Marshal.AllocHGlobal((nint)(bytes + alignment + (nuint)sizeof(nint)));
            nuint aligned = ((nuint)raw + (nuint)sizeof(nint) + alignment - 1) & ~(alignment - 1);
            ((nint*)aligned)[-1] = raw;
            return (void*)aligned;
        }

        public static void AlignedFree(void* p)
        {
            if (p != null) Marshal.FreeHGlobal(((nint*)p)[-1]);
        }
    }
}
