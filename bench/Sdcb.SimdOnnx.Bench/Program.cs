using System.Diagnostics;
using System.Reflection;
using G = Sdcb.SimdOnnx;
using P = Sdcb.SimdPaddleOCR.OnnxSharp;

// SimdCore spike: run a PP-OCR det model through the GcdrNet engine and SimdPaddleOCR,
// on CPU and Vulkan; compare outputs (against the first engine listed), latency and memory.
// usage: <model.onnx> <input_HxW.f32> <engines: gcdr,paddle,gcdr-vk,paddle-vk,gcdr-mtl,paddle-mtl> [threads] [iters]
if (args[0] == "--micro")
{
    Environment.SetEnvironmentVariable("GCDRNET_THREADS", "1");
    Sdcb.SimdOnnx.Bench.Micro.Run(args[1..]);
    return;
}
string modelPath = args[0], inputPath = args[1];
string[] engines = args[2].Split(',');
int threads = args.Length > 3 ? int.Parse(args[3]) : Environment.ProcessorCount;
int iters = args.Length > 4 ? int.Parse(args[4]) : 20;
bool prof = Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1";
// must be set before the GcdrNet kernels' static init
Environment.SetEnvironmentVariable("GCDRNET_THREADS", threads.ToString());

string[] hw = Path.GetFileNameWithoutExtension(inputPath).Split('_')[^1].Split(':')[^1].Split('x');
int[] shape = hw.Length == 4 ? hw.Select(int.Parse).ToArray() : [1, 3, int.Parse(hw[0]), int.Parse(hw[1])];
float[] input = new float[shape.Aggregate(1, (a, b) => a * b)];
if (inputPath.StartsWith("rand:"))
{
    // deterministic smooth pseudo-image in [-1, 1] (rec/cls inputs are normalized that way)
    var rng = new Random(42);
    for (int i = 0; i < input.Length; i++) input[i] = MathF.Sin(i * 0.013f) * 0.6f + (float)(rng.NextDouble() - 0.5) * 0.8f;
}
else
{
    using FileStream fs = File.OpenRead(inputPath);
    fs.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()));
}
byte[] modelBytes = File.ReadAllBytes(modelPath);
if (Environment.GetEnvironmentVariable("BENCH_SAVE_INPUT") is { Length: > 0 } saveIn)
    File.WriteAllBytes(saveIn, System.Runtime.InteropServices.MemoryMarshal.AsBytes(input.AsSpan()).ToArray());

long ws0 = Process.GetCurrentProcess().WorkingSet64;
Console.WriteLine($"model={Path.GetFileName(modelPath)} input={string.Join('x', shape)} threads={threads} iters={iters}");

float[]? reference = null;
int[]? outShape = null;
foreach (string e in engines)
{
    float[] o = e switch
    {
        _ when e.StartsWith("file:") => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(e[5..])).ToArray(),
        "gcdr" => RunGcdr(),
        "gcdr-compile" => CompileSweep(),
        "gcdr-ref" => RunGcdr(new G.Engine.CompileOptions(false)),
        "paddle" => RunPaddle(),
        "gcdr-vk" => RunGcdrVk(),
        "paddle-vk" => RunPaddleGpu("Vulkan", "GpuBackend", "GpuSession"),
        "gcdr-mtl" => RunGcdrMtl(),
        "paddle-mtl" => RunPaddleGpu("Metal", "MetalBackend", "MetalSession"),
        _ => throw new ArgumentException(e),
    };
    if (reference is null) { reference = o; continue; }
    if (o.Length != reference.Length) { Console.WriteLine($"  vs {engines[0]}: LENGTH {o.Length} != {reference.Length}"); continue; }
    double maxAbs = 0, sumAbs = 0; int flips = 0;
    for (int i = 0; i < o.Length; i++)
    {
        double d = Math.Abs(o[i] - reference[i]);
        maxAbs = Math.Max(maxAbs, d); sumAbs += d;
        if (o[i] > 0.3f != reference[i] > 0.3f) flips++;
    }
    string argmax = "";
    if (outShape is { Length: >= 2 } os && os[^1] > 1)
    {
        int c = os[^1], rows = o.Length / c, diff = 0;
        for (int r = 0; r < rows; r++)
        {
            int ia = 0, ib = 0;
            for (int k = 1; k < c; k++)
            {
                if (o[r * c + k] > o[r * c + ia]) ia = k;
                if (reference[r * c + k] > reference[r * c + ib]) ib = k;
            }
            if (ia != ib) diff++;
        }
        argmax = $" argmax-diff={diff}/{rows}";
    }
    Console.WriteLine($"  vs {engines[0]}: max|d|={maxAbs:E2} mean|d|={sumAbs / o.Length:E2} mask@0.3 flips={flips} / {o.Length}{argmax}");
}

float[] CompileSweep()
{
    // per-shape specialization cost once the model is loaded (REC widths vary per line)
    G.Onnx.OnnxModel m = G.Onnx.OnnxModel.Load(modelBytes);
    G.Engine.CompiledModel.Compile(m, shape);
    var sw = Stopwatch.StartNew();
    int count = 0;
    for (int w = shape[3] + 8; w < shape[3] + 8 + 32 * 8; w += 32, count++)
        G.Engine.CompiledModel.Compile(m, [shape[0], shape[1], shape[2], w]);
    double compile = sw.Elapsed.TotalMilliseconds / count;
    sw.Restart();
    for (int w = shape[3] + 8; w < shape[3] + 8 + 32 * 8; w += 32)
        G.Engine.CompiledModel.Compile(m, [shape[0], shape[1], shape[2], w]).CreateSession().Dispose();
    Console.WriteLine($"compile: {compile:F2} ms, compile+pack+session: {sw.Elapsed.TotalMilliseconds / count:F2} ms per new shape");
    return RunGcdr();
}

float[] RunGcdr(G.Engine.CompileOptions? opt = null)
{
    var sw = Stopwatch.StartNew();
    G.Onnx.OnnxModel m = G.Onnx.OnnxModel.Load(modelBytes);
    G.Engine.CompiledModel cm = opt is null ? G.Engine.CompiledModel.Compile(m, shape) : G.Engine.CompiledModel.Compile(m, shape, opt);
    using var s = cm.CreateSession();
    double compileMs = sw.Elapsed.TotalMilliseconds;
    input.CopyTo(s.Input);
    double first = Time(() => s.Run());
    s.ResetProfile();
    var times = Enumerable.Range(0, iters).Select(_ => { input.CopyTo(s.Input); return Time(() => s.Run()); }).ToArray();
    Report("gcdr", compileMs, first, times, $"nodes={cm.NodeCount} arena={cm.ArenaFloats * 4.0 / 1048576:F1}MB");
    outShape = s.OutputShape;
    if (prof)
    {
        foreach (var (op, t) in s.Profile().OrderByDescending(kv => kv.Value))
            Console.WriteLine($"    {op,-14} {t.TotalMilliseconds / iters,7:F2}ms");
        string Cls(G.Engine.ExecNode n) => n.P is G.Engine.ConvP c
            ? $"{(c.Group > 1 ? "dw" : "dense")}{c.Kernel[0]}x{c.Kernel[1]}s{c.Strides[0]}"
            : n.Op.ToString();
        foreach (var g in s.NodeProfile().Where(x => x.n.Op == G.Engine.ExecOp.Conv).GroupBy(x => Cls(x.n)).OrderByDescending(g => g.Sum(x => x.t.Ticks)))
            Console.WriteLine($"    conv {g.Key,-14} x{g.Count(),-3} {g.Sum(x => x.t.TotalMilliseconds) / iters,7:F2}ms");
        if (Environment.GetEnvironmentVariable("BENCH_TOP") is { } top)
            foreach (var (i, n, t) in s.NodeProfile()
                .Where(x => Environment.GetEnvironmentVariable("BENCH_OP") is not { } only || x.n.Op.ToString() == only)
                .OrderByDescending(x => x.t).Take(int.Parse(top)))
                Console.WriteLine($"      #{i,-3} {n.Op,-12} {t.TotalMilliseconds / iters,6:F3}ms " +
                    string.Join(" ", n.In.Select(v => $"[{string.Join('x', cm.Plan.Values[v].PhysShape)}]")) + " -> " +
                    string.Join(" ", n.Out.Select(v => $"[{string.Join('x', cm.Plan.Values[v].PhysShape)}]")) + $" {n.P}");
    }
    return s.Output.ToArray();
}

float[] RunGcdrVk()
{
    var sw = Stopwatch.StartNew();
    G.Onnx.OnnxModel m = G.Onnx.OnnxModel.Load(modelBytes);
    var be = G.Backends.Vulkan.VkBackend.Default;
    using var s = be.NewSession(m, shape);
    double compileMs = sw.Elapsed.TotalMilliseconds;
    input.CopyTo(s.Input);
    double first = Time(() => s.Run());
    var times = Enumerable.Range(0, iters).Select(_ => { input.CopyTo(s.Input); return Time(() => s.Run()); }).ToArray();
    Report("gcdr-vk", compileMs, first, times, $"f16={s.F16} gpu-arena={s.Arena.Size / 1048576.0:F1}MB");
    float[] o = s.Output.ToArray();
    if (prof)
        foreach (var (name, ms) in s.DebugProfile().OrderByDescending(kv => kv.Value))
            Console.WriteLine($"    {name,-16} {ms,7:F3}ms");
    return o;
}

float[] RunGcdrMtl()
{
    var sw = Stopwatch.StartNew();
    G.Onnx.OnnxModel m = G.Onnx.OnnxModel.Load(modelBytes);
    var be = G.Backends.Metal.MetalBackend.Default;
    using var s = be.NewSession(m, shape);
    double compileMs = sw.Elapsed.TotalMilliseconds;
    input.CopyTo(s.Input);
    double first = Time(() => s.Run());
    var times = Enumerable.Range(0, iters).Select(_ => { input.CopyTo(s.Input); return Time(() => s.Run()); }).ToArray();
    Report("gcdr-mtl", compileMs, first, times, $"gpu-arena={s.DebugPlan.ArenaFloats * 2.0 / 1048576:F1}MB");
    float[] o = s.Output.ToArray();
    if (prof)
        foreach (var (name, ms) in s.DebugProfile().OrderByDescending(kv => kv.Value))
            Console.WriteLine($"    {name,-16} {ms,7:F3}ms");
    return o;
}

float[] RunPaddle()
{
    var sw = Stopwatch.StartNew();
    P.Model m = P.Model.Load(modelBytes);
    using var s = new P.InferenceSession(m, shape, Math.Min(threads, 16));
    double compileMs = sw.Elapsed.TotalMilliseconds;
    float[] o = [];
    double first = Time(() => o = s.Run(input));
    var ist = typeof(P.InferenceSession);
    const BindingFlags NP = BindingFlags.NonPublic | BindingFlags.Static;
    if (prof) ist.GetMethod("EnableProfiling", NP)!.Invoke(null, [true]);
    var times = Enumerable.Range(0, iters).Select(_ => Time(() => o = s.Run(input))).ToArray();
    Report("paddle", compileMs, first, times, "");
    if (prof)
    {
        var snap = ((long Ticks, long Calls)[])ist.GetMethod("ProfileSnapshot", NP)!.Invoke(null, null)!;
        for (int i = 0; i < snap.Length; i++)
            if (snap[i].Ticks > 0)
                Console.WriteLine($"    {(P.OperatorId)i,-14} {snap[i].Ticks * 1000.0 / Stopwatch.Frequency / iters,7:F2}ms");
        var conv = ((long Ticks, long Calls)[])ist.GetMethod("ConvClassProfileSnapshot", NP)!.Invoke(null, null)!;
        string[] convNames = ["Conv1x1", "Conv3x3", "Depthwise3x3", "Stride2Conv3x3", "OtherConv"];
        for (int i = 0; i < conv.Length && i < convNames.Length; i++)
            if (conv[i].Ticks > 0)
                Console.WriteLine($"    conv {convNames[i],-14} x{conv[i].Calls / iters,-3} {conv[i].Ticks * 1000.0 / Stopwatch.Frequency / iters,7:F2}ms");
        ist.GetMethod("EnableProfiling", NP)!.Invoke(null, [false]);
    }
    return o;
}

float[] RunPaddleGpu(string backend, string factory, string sessionType)
{
    var sw = Stopwatch.StartNew();
    P.Model m = P.Model.Load(modelBytes);
    using var cpu = new P.InferenceSession(m, shape, 1);
    Assembly asm = typeof(P.Model).Assembly;
    object sess = asm.GetType($"Sdcb.SimdPaddleOCR.Backends.{backend}.{factory}")!
        .GetMethod("CreateSession", BindingFlags.NonPublic | BindingFlags.Static)!
        .Invoke(null, [cpu.CompiledModel, Enum.Parse<Sdcb.SimdPaddleOCR.OcrBackend>(backend)])!;
    if (sess.GetType().Name != sessionType) throw new InvalidOperationException($"paddle {backend}: got {sess.GetType().Name}");
    Type ios = asm.GetType("Sdcb.SimdPaddleOCR.OnnxSharp.IOcrSession")!;
    var reshape = (ReshapeFn)Delegate.CreateDelegate(typeof(ReshapeFn), sess, ios.GetMethod("Reshape")!);
    var run = (RunFn)Delegate.CreateDelegate(typeof(RunFn), sess, ios.GetMethod("RunInternal")!);
    reshape(shape);
    double compileMs = sw.Elapsed.TotalMilliseconds;
    float[] o = [];
    double first = Time(() => o = run(input).ToArray());
    var times = Enumerable.Range(0, iters).Select(_ => Time(() => o = run(input).ToArray())).ToArray();
    bool alive = (bool)ios.GetProperty("GpuAlive")!.GetValue(sess)!;
    Report($"paddle-{(backend == "Metal" ? "mtl" : "vk")}", compileMs, first, times, $"gpuAlive={alive}");
    ((IDisposable)sess).Dispose();
    return o;
}

static double Time(Action a)
{
    long t0 = Stopwatch.GetTimestamp();
    a();
    return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
}

void Report(string name, double compileMs, double first, double[] t, string extra)
{
    Array.Sort(t);
    var p = Process.GetCurrentProcess();
    p.Refresh();
    Console.WriteLine($"{name,-9} load+compile={compileMs,7:F1}ms first={first,7:F1}ms median={t[t.Length / 2],7:F2}ms min={t[0],7:F2}ms " +
        $"peakWS={(p.PeakWorkingSet64 - ws0) / 1048576.0,6:F1}MB(+base) private={p.PrivateMemorySize64 / 1048576.0,6:F1}MB {extra}");
}

delegate void ReshapeFn(ReadOnlySpan<int> shape);
delegate ReadOnlySpan<float> RunFn(ReadOnlySpan<float> input);
