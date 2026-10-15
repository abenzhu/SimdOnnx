using System.IO.Compression;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>Minimal PNG reader for the golden fixtures: 8-bit truecolor (color type 2), no interlace.</summary>
internal static class Png
{
    public static byte[] DecodeRgb8(byte[] png, out int width, out int height)
    {
        int pos = 8;
        width = height = 0;
        var idat = new MemoryStream();
        while (pos < png.Length)
        {
            int len = BigEndian(png, pos);
            string type = System.Text.Encoding.ASCII.GetString(png, pos + 4, 4);
            if (type == "IHDR")
            {
                width = BigEndian(png, pos + 8);
                height = BigEndian(png, pos + 12);
                if (png[pos + 16] != 8 || png[pos + 17] != 2 || png[pos + 20] != 0)
                    throw new NotSupportedException("PNG: only 8-bit RGB, non-interlaced");
            }
            else if (type == "IDAT") idat.Write(png, pos + 8, len);
            else if (type == "IEND") break;
            pos += 12 + len;
        }
        // zlib stream: skip the 2-byte header, inflate the deflate body
        idat.Position = 2;
        using var z = new DeflateStream(idat, CompressionMode.Decompress);
        int stride = width * 3;
        byte[] raw = new byte[(stride + 1) * height];
        int got = 0;
        while (got < raw.Length)
        {
            int r = z.Read(raw, got, raw.Length - got);
            if (r <= 0) throw new InvalidDataException("PNG: truncated image data");
            got += r;
        }
        byte[] img = new byte[stride * height];
        for (int y = 0; y < height; y++)
        {
            int f = raw[y * (stride + 1)], src = y * (stride + 1) + 1, dst = y * stride;
            for (int x = 0; x < stride; x++)
            {
                int a = x >= 3 ? img[dst + x - 3] : 0, b = y > 0 ? img[dst - stride + x] : 0;
                int c = x >= 3 && y > 0 ? img[dst - stride + x - 3] : 0;
                int v = raw[src + x];
                img[dst + x] = (byte)(f switch
                {
                    0 => v,
                    1 => v + a,
                    2 => v + b,
                    3 => v + (a + b) / 2,
                    4 => v + Paeth(a, b, c),
                    _ => throw new InvalidDataException($"PNG: filter {f}"),
                });
            }
        }
        return img;
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    private static int BigEndian(byte[] b, int i) => b[i] << 24 | b[i + 1] << 16 | b[i + 2] << 8 | b[i + 3];
}
