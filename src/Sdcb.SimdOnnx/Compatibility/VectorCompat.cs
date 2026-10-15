// netstandard2.0 only (see the csproj): Vector<T> element access that System.Numerics.Vectors lacks.
using System.Runtime.CompilerServices;

namespace System.Numerics
{
    internal static class VectorCompat
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T GetElement<T>(this Vector<T> vector, int index) where T : struct =>
            Unsafe.Add(ref Unsafe.As<Vector<T>, T>(ref vector), index);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<T> WithElement<T>(this Vector<T> vector, int index, T value) where T : struct
        {
            Unsafe.Add(ref Unsafe.As<Vector<T>, T>(ref vector), index) = value;
            return vector;
        }
    }
}
