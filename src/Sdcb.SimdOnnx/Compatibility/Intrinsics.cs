// netstandard2.0 only (see the csproj): compile-time stand-ins for the System.Runtime.Intrinsics
// surface the kernels name. Every IsSupported / IsHardwareAccelerated is false, so the hardware tiers
// are never entered and the Vector<T> / scalar paths run; the members exist only so the shared
// kernel sources compile unchanged. Calling any of them is a bug and throws.
#pragma warning disable CS1591, IDE0060
using System.Runtime.CompilerServices;

namespace System.Runtime.Intrinsics
{
    internal static class NotSupported
    {
        public static Exception Ex() => new PlatformNotSupportedException("Hardware intrinsics are unavailable on netstandard2.0.");
    }

    internal readonly struct Vector128<T> where T : struct
    {
        public static Vector128<T> Zero => default;
        public T this[int i] => throw NotSupported.Ex();
        public static Vector128<T> One => throw NotSupported.Ex();
        public static Vector128<T> AllBitsSet => throw NotSupported.Ex();
        public static int Count => 16 / Unsafe.SizeOf<T>();
        public static Vector128<T> operator +(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator -(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator *(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator *(Vector128<T> a, T b) => throw NotSupported.Ex();
        public static Vector128<T> operator *(T a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator /(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator /(Vector128<T> a, T b) => throw NotSupported.Ex();
        public static Vector128<T> operator &(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator |(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator ^(Vector128<T> a, Vector128<T> b) => throw NotSupported.Ex();
        public static Vector128<T> operator ~(Vector128<T> a) => throw NotSupported.Ex();
        public static Vector128<T> operator -(Vector128<T> a) => throw NotSupported.Ex();
        public static Vector128<T> operator <<(Vector128<T> a, int b) => throw NotSupported.Ex();
        public static Vector128<T> operator >>(Vector128<T> a, int b) => throw NotSupported.Ex();
    }

    internal readonly struct Vector256<T> where T : struct
    {
        public static Vector256<T> Zero => default;
        public T this[int i] => throw NotSupported.Ex();
        public static Vector256<T> One => throw NotSupported.Ex();
        public static Vector256<T> AllBitsSet => throw NotSupported.Ex();
        public static int Count => 32 / Unsafe.SizeOf<T>();
        public static Vector256<T> operator +(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator -(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator *(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator *(Vector256<T> a, T b) => throw NotSupported.Ex();
        public static Vector256<T> operator *(T a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator /(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator /(Vector256<T> a, T b) => throw NotSupported.Ex();
        public static Vector256<T> operator &(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator |(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator ^(Vector256<T> a, Vector256<T> b) => throw NotSupported.Ex();
        public static Vector256<T> operator ~(Vector256<T> a) => throw NotSupported.Ex();
        public static Vector256<T> operator -(Vector256<T> a) => throw NotSupported.Ex();
        public static Vector256<T> operator <<(Vector256<T> a, int b) => throw NotSupported.Ex();
        public static Vector256<T> operator >>(Vector256<T> a, int b) => throw NotSupported.Ex();
    }

    internal readonly struct Vector512<T> where T : struct
    {
        public static Vector512<T> Zero => default;
        public T this[int i] => throw NotSupported.Ex();
        public static Vector512<T> One => throw NotSupported.Ex();
        public static int Count => 64 / Unsafe.SizeOf<T>();
        public static Vector512<T> operator +(Vector512<T> a, Vector512<T> b) => throw NotSupported.Ex();
        public static Vector512<T> operator -(Vector512<T> a, Vector512<T> b) => throw NotSupported.Ex();
        public static Vector512<T> operator *(Vector512<T> a, Vector512<T> b) => throw NotSupported.Ex();
        public static Vector512<T> operator *(Vector512<T> a, T b) => throw NotSupported.Ex();
    }

    internal static unsafe class Vector128
    {
        public static bool IsHardwareAccelerated => false;
        public static Vector128<T> Create<T>(T v) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Create<T>(T a, T b, T c, T d) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Create<T>(T a, T b, T c, T d, T e, T f, T g, T h) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Load<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static Vector128<T> LoadUnsafe<T>(ref T p) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> LoadUnsafe<T>(ref T p, nuint i) where T : struct => throw NotSupported.Ex();
        public static void Store<T>(this Vector128<T> v, T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void StoreUnsafe<T>(this Vector128<T> v, ref T p) where T : struct => throw NotSupported.Ex();
        public static void StoreUnsafe<T>(this Vector128<T> v, ref T p, nuint i) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Add<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Subtract<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Multiply<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Divide<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Negate<T>(Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Max<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Min<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Abs<T>(Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> AndNot<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> ConditionalSelect<T>(Vector128<T> m, Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> LessThan<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> GreaterThan<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> ShiftLeft<T>(Vector128<T> a, int n) where T : struct => throw NotSupported.Ex();
        public static Vector128<float> FusedMultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c) => throw NotSupported.Ex();
        public static Vector128<float> Floor(Vector128<float> a) => throw NotSupported.Ex();
        public static Vector128<float> Sqrt(Vector128<float> a) => throw NotSupported.Ex();
        public static Vector128<int> ConvertToInt32(Vector128<float> a) => throw NotSupported.Ex();
        public static Vector128<float> ConvertToSingle(Vector128<int> a) => throw NotSupported.Ex();
        public static T Sum<T>(Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static T ToScalar<T>(this Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static T GetElement<T>(this Vector128<T> a, int i) where T : struct => throw NotSupported.Ex();
        public static Vector128<float> AsSingle<T>(this Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<int> AsInt32<T>(this Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<uint> AsUInt32<T>(this Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<byte> AsByte<T>(this Vector128<T> a) where T : struct => throw NotSupported.Ex();
    }

    internal static unsafe class Vector256
    {
        public static bool IsHardwareAccelerated => false;
        public static Vector256<T> Create<T>(T v) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Create<T>(T a, T b, T c, T d, T e, T f, T g, T h) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Create<T>(Vector128<T> lo, Vector128<T> hi) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Load<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static Vector256<T> LoadUnsafe<T>(ref T p) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> LoadUnsafe<T>(ref T p, nuint i) where T : struct => throw NotSupported.Ex();
        public static void Store<T>(this Vector256<T> v, T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void StoreUnsafe<T>(this Vector256<T> v, ref T p) where T : struct => throw NotSupported.Ex();
        public static void StoreUnsafe<T>(this Vector256<T> v, ref T p, nuint i) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Add<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Subtract<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Multiply<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Divide<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Negate<T>(Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Max<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Min<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Abs<T>(Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> AndNot<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> ConditionalSelect<T>(Vector256<T> m, Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> LessThan<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> GreaterThan<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> ShiftLeft<T>(Vector256<T> a, int n) where T : struct => throw NotSupported.Ex();
        public static Vector256<float> FusedMultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c) => throw NotSupported.Ex();
        public static Vector256<float> Floor(Vector256<float> a) => throw NotSupported.Ex();
        public static Vector256<float> Sqrt(Vector256<float> a) => throw NotSupported.Ex();
        public static Vector256<int> ConvertToInt32(Vector256<float> a) => throw NotSupported.Ex();
        public static Vector256<float> ConvertToSingle(Vector256<int> a) => throw NotSupported.Ex();
        public static T Sum<T>(Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static T ToScalar<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static T GetElement<T>(this Vector256<T> a, int i) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> GetLower<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> GetUpper<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector256<float> AsSingle<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector256<int> AsInt32<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector256<uint> AsUInt32<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector256<byte> AsByte<T>(this Vector256<T> a) where T : struct => throw NotSupported.Ex();
    }

    internal static unsafe class Vector512
    {
        public static bool IsHardwareAccelerated => false;
        public static Vector512<T> Create<T>(T v) where T : struct => throw NotSupported.Ex();
        public static Vector512<T> Load<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void Store<T>(this Vector512<T> v, T* p) where T : unmanaged => throw NotSupported.Ex();
        public static Vector512<T> Max<T>(Vector512<T> a, Vector512<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector512<T> Min<T>(Vector512<T> a, Vector512<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector512<float> FusedMultiplyAdd(Vector512<float> a, Vector512<float> b, Vector512<float> c) => throw NotSupported.Ex();
    }
}

namespace System.Runtime.Intrinsics.X86
{
    internal abstract unsafe class Sse
    {
        public static bool IsSupported => false;
    }

    internal abstract unsafe class Avx : Sse
    {
        public static new bool IsSupported => false;
        public static Vector256<T> LoadVector256<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void Store<T>(T* p, Vector256<T> v) where T : unmanaged => throw NotSupported.Ex();
        public static Vector256<float> BroadcastScalarToVector256(float* p) => throw NotSupported.Ex();
        public static Vector256<float> MaskLoad(float* p, Vector256<float> m) => throw NotSupported.Ex();
        public static void MaskStore(float* p, Vector256<float> m, Vector256<float> v) => throw NotSupported.Ex();
        public static Vector256<T> Add<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Subtract<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Multiply<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Divide<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Max<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Min<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> And<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> AndNot<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Or<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Xor<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> Permute2x128<T>(Vector256<T> a, Vector256<T> b, byte c) where T : struct => throw NotSupported.Ex();
        public static Vector256<float> Shuffle(Vector256<float> a, Vector256<float> b, byte c) => throw NotSupported.Ex();
        public static Vector256<float> UnpackHigh(Vector256<float> a, Vector256<float> b) => throw NotSupported.Ex();
        public static Vector256<float> UnpackLow(Vector256<float> a, Vector256<float> b) => throw NotSupported.Ex();
        public static Vector256<int> ConvertToVector256Int32(Vector256<float> a) => throw NotSupported.Ex();
        public static Vector256<float> ConvertToVector256Single(Vector256<int> a) => throw NotSupported.Ex();
        public static Vector256<float> Floor(Vector256<float> a) => throw NotSupported.Ex();
    }

    internal abstract unsafe class Avx2 : Avx
    {
        public static new bool IsSupported => false;
        public static Vector256<T> Add<T>(Vector256<T> a, Vector256<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector256<T> BroadcastScalarToVector256<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static Vector256<int> ConvertToVector256Int32(byte* p) => throw NotSupported.Ex();
        public static Vector256<int> ConvertToVector256Int32(Vector128<byte> v) => throw NotSupported.Ex();
        public static Vector256<float> GatherVector256(float* p, Vector256<int> i, byte scale) => throw NotSupported.Ex();
        public static Vector256<float> PermuteVar8x32(Vector256<float> a, Vector256<int> i) => throw NotSupported.Ex();
        public static Vector256<int> PermuteVar8x32(Vector256<int> a, Vector256<int> i) => throw NotSupported.Ex();
        public static Vector256<int> ShiftLeftLogical(Vector256<int> a, byte n) => throw NotSupported.Ex();
    }

    internal abstract unsafe class Fma : Avx
    {
        public static new bool IsSupported => false;
        public static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c) => throw NotSupported.Ex();
        public static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c) => throw NotSupported.Ex();
    }

    internal abstract unsafe class Avx512F : Avx2
    {
        public static new bool IsSupported => false;
        public static Vector512<T> LoadVector512<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void Store<T>(T* p, Vector512<T> v) where T : unmanaged => throw NotSupported.Ex();
        public static Vector512<T> Add<T>(Vector512<T> a, Vector512<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector512<T> Max<T>(Vector512<T> a, Vector512<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector512<float> FusedMultiplyAdd(Vector512<float> a, Vector512<float> b, Vector512<float> c) => throw NotSupported.Ex();
    }
}

namespace System.Runtime.Intrinsics.Arm
{
    internal static unsafe class AdvSimd
    {
        public static bool IsSupported => false;
        public static Vector128<T> LoadVector128<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static Vector128<T> LoadAndReplicateToVector128<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void Store<T>(T* p, Vector128<T> v) where T : unmanaged => throw NotSupported.Ex();
        public static Vector128<T> Add<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Subtract<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Multiply<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Max<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Min<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<float> FusedMultiplyAdd(Vector128<float> acc, Vector128<float> a, Vector128<float> b) => throw NotSupported.Ex();

        internal static class Arm64
        {
            public static bool IsSupported => false;
            public static Vector128<float> FusedMultiplyAddBySelectedScalar(Vector128<float> acc, Vector128<float> a, Vector128<float> b, byte i) => throw NotSupported.Ex();
        }
    }
}

namespace System.Runtime.Intrinsics.Wasm
{
    internal static unsafe class PackedSimd
    {
        public static bool IsSupported => false;
        public static Vector128<T> LoadVector128<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static Vector128<T> LoadScalarAndSplatVector128<T>(T* p) where T : unmanaged => throw NotSupported.Ex();
        public static void Store<T>(T* p, Vector128<T> v) where T : unmanaged => throw NotSupported.Ex();
        public static Vector128<T> Splat<T>(T v) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Add<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Subtract<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Multiply<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Divide<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Negate<T>(Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> Abs<T>(Vector128<T> a) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> And<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> PseudoMax<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> PseudoMin<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> CompareGreaterThan<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> CompareLessThan<T>(Vector128<T> a, Vector128<T> b) where T : struct => throw NotSupported.Ex();
        public static Vector128<T> ShiftLeft<T>(Vector128<T> a, int n) where T : struct => throw NotSupported.Ex();
        public static Vector128<int> ConvertToInt32Saturate(Vector128<float> a) => throw NotSupported.Ex();
        public static Vector128<float> ConvertToSingle(Vector128<int> a) => throw NotSupported.Ex();
    }
}
