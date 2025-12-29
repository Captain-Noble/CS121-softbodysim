using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public sealed class PBDWorld : MonoBehaviour
{
    public enum ExecutionMode { Serial, Parallel }

    private static readonly double TickToMs = 1000.0 / Stopwatch.Frequency;



    [Header("Execution")]
    public ExecutionMode executionMode = ExecutionMode.Parallel;
    public int maxWorkerThreads = 0;
    public int rangeChunk = 4096;

    [Header("Time stepping")]
    public bool simulateInFixedUpdate = true;
    public float fixedDt = 1f / 60f;
    [Range(1, 8)] public int substeps = 2;

    [Header("Gravity / Ground")]
    public Vector3 gravity = new Vector3(0, -9.81f, 0);
    public bool groundEnabled = true;
    public float groundY = 0f;

    [Header("Ground friction")]
    public bool groundFrictionEnabled = true;
    [Range(0f, 5f)] public float groundMuStatic = 0.8f;
    [Range(0f, 5f)] public float groundMuDynamic = 0.6f;
    public float groundContactEps = 1e-4f;
    public float groundSlop = 1e-4f;

    [Header("Optional contact damping")]
    [Range(0f, 50f)] public float groundTangentDamping = 0.0f;

    [Header("Solver")]
    [Range(1, 30)] public int solverIterations = 6;
    [Range(0.2f, 2.5f)] public float omega = 1.6f;

    [Header("Profiling")]
    public bool printStats = true;
    public float statsPeriodSeconds = 1f;

    private readonly List<PBDSoftBody> _bodies = new List<PBDSoftBody>(64);
    private bool _dirtyRebuild = true;

    private Vector3[] _x, _v, _xStar;
    private float[] _w;

    private int[] _edgeI0, _edgeI1;
    private float[] _edgeRest, _edgeCompliance, _edgeLambda;

    private int[] _tetA, _tetB, _tetC, _tetD;
    private float[] _tetRestVol, _tetCompliance, _tetLambda;

    private Vector3[] _mergedDelta;
    private int[] _mergedCount, _mergedTouched;
    private int _mergedTouchedCount;

    private ParallelOptions _opt = new ParallelOptions();

    private ThreadLocal<PBDKernels.LocalAccum> _tlsEdge;
    private ThreadLocal<PBDKernels.LocalAccum> _tlsTet;

    private readonly object _edgeRegLock = new object();
    private readonly List<PBDKernels.LocalAccum> _edgeRegistered = new List<PBDKernels.LocalAccum>(64);
    private PBDKernels.LocalAccum[] _edgeSnapshot = Array.Empty<PBDKernels.LocalAccum>();

    private readonly object _tetRegLock = new object();
    private readonly List<PBDKernels.LocalAccum> _tetRegistered = new List<PBDKernels.LocalAccum>(64);
    private PBDKernels.LocalAccum[] _tetSnapshot = Array.Empty<PBDKernels.LocalAccum>();

    private readonly Stopwatch _sw = new Stopwatch();
    private float _statsTimer;
    private int _statsFrames;

    private struct Timings
    {
        public double totalMs, rebuildMs, predictMs;
        public double edgeComputeMs, edgeMergeMs, applyEdgeMs;
        public double tetComputeMs, tetMergeMs, applyTetMs;
        public double groundMs, commitMs, uploadMs;
        public void Clear() => this = default;
    }
    private Timings _period;

    private int EffectiveThreadCount()
    {
        int tc = (maxWorkerThreads > 0) ? maxWorkerThreads : Environment.ProcessorCount;
        return Mathf.Max(1, tc);
    }

    private bool UseParallel => executionMode == ExecutionMode.Parallel && EffectiveThreadCount() > 1;

    private void Awake()
    {
        ApplyThreadCount();
    }

    private void OnEnable() => _dirtyRebuild = true;

    private void Update()
    {
        if (!simulateInFixedUpdate) Step(Time.deltaTime);
        TickStats();
    }

    private void FixedUpdate()
    {
        if (!simulateInFixedUpdate) return;
        Step(fixedDt > 0f ? fixedDt : Time.fixedDeltaTime);
    }

    public void RegisterBody(PBDSoftBody body)
    {
        if (body == null) return;
        if (!_bodies.Contains(body))
        {
            _bodies.Add(body);
            _dirtyRebuild = true;
        }
    }

    public void UnregisterBody(PBDSoftBody body)
    {
        if (body == null) return;
        if (_bodies.Remove(body)) _dirtyRebuild = true;
    }

    private void Step(float dt)
    {
        if (_bodies.Count == 0) return;

        _sw.Restart();

        if (_dirtyRebuild)
        {
            double t0 = _sw.Elapsed.TotalMilliseconds;
            RebuildWorld();
            _period.rebuildMs += _sw.Elapsed.TotalMilliseconds - t0;
        }

        if (_x == null || _x.Length == 0) return;

        ApplyThreadCount();
        EnsureTLS();

        bool parallel = UseParallel;

        int ss = Mathf.Max(1, substeps);
        float sdt = dt / ss;

        for (int step = 0; step < ss; step++)
        {
            {
                double t0 = _sw.Elapsed.TotalMilliseconds;
                PBDKernels.Predict(_x, _v, _w, _xStar, gravity, sdt, parallel, _opt);
                _period.predictMs += _sw.Elapsed.TotalMilliseconds - t0;
            }

            float invDt = sdt > 1e-12f ? 1f / sdt : 0f;

            for (int it = 0; it < solverIterations; it++)
            {
                {
                    if (parallel)
                    {
                        int resetCount = SnapshotAccums(_edgeRegistered, _edgeRegLock, ref _edgeSnapshot);
                        PBDKernels.ResetRegisteredAccums(_edgeSnapshot, resetCount);
                        PBDKernels.ResetMerged(_mergedDelta, _mergedCount, _mergedTouched, ref _mergedTouchedCount);

                        long t0 = Stopwatch.GetTimestamp();
                        PBDKernels.SolveEdgesComputeParallel(
                            _edgeI0, _edgeI1, _edgeRest, _edgeCompliance, _edgeLambda,
                            _xStar, _w, sdt,
                            _tlsEdge, _opt);
                        long t1 = Stopwatch.GetTimestamp();

                        int mergeCount = SnapshotAccums(_edgeRegistered, _edgeRegLock, ref _edgeSnapshot);
                        PBDKernels.MergeRegisteredAccums(_edgeSnapshot, mergeCount, _mergedDelta, _mergedCount, _mergedTouched, ref _mergedTouchedCount);
                        long t2 = Stopwatch.GetTimestamp();

                        _period.edgeComputeMs += (t1 - t0) * TickToMs;
                        _period.edgeMergeMs += (t2 - t1) * TickToMs;
                    }
                    else
                    {
                        double t0 = _sw.Elapsed.TotalMilliseconds;
                        SolveEdgesSerial(sdt);
                        _period.edgeComputeMs += _sw.Elapsed.TotalMilliseconds - t0;
                    }

                    double tApply = _sw.Elapsed.TotalMilliseconds;
                    PBDKernels.ApplyMergedDelta(_xStar, _mergedDelta, _mergedCount, _mergedTouched, _mergedTouchedCount, omega, parallel, _opt);
                    _period.applyEdgeMs += _sw.Elapsed.TotalMilliseconds - tApply;
                }

                {
                    if (parallel)
                    {
                        int resetCount = SnapshotAccums(_tetRegistered, _tetRegLock, ref _tetSnapshot);
                        PBDKernels.ResetRegisteredAccums(_tetSnapshot, resetCount);
                        PBDKernels.ResetMerged(_mergedDelta, _mergedCount, _mergedTouched, ref _mergedTouchedCount);

                        long t0 = Stopwatch.GetTimestamp();
                        PBDKernels.SolveTetsComputeParallel(
                            _tetA, _tetB, _tetC, _tetD, _tetRestVol, _tetCompliance, _tetLambda,
                            _xStar, _w, sdt,
                            _tlsTet, _opt);
                        long t1 = Stopwatch.GetTimestamp();

                        int mergeCount = SnapshotAccums(_tetRegistered, _tetRegLock, ref _tetSnapshot);
                        PBDKernels.MergeRegisteredAccums(_tetSnapshot, mergeCount, _mergedDelta, _mergedCount, _mergedTouched, ref _mergedTouchedCount);
                        long t2 = Stopwatch.GetTimestamp();

                        _period.tetComputeMs += (t1 - t0) * TickToMs;
                        _period.tetMergeMs += (t2 - t1) * TickToMs;
                    }
                    else
                    {
                        double t0 = _sw.Elapsed.TotalMilliseconds;
                        SolveTetsSerial(sdt);
                        _period.tetComputeMs += _sw.Elapsed.TotalMilliseconds - t0;
                    }

                    double tApply = _sw.Elapsed.TotalMilliseconds;
                    PBDKernels.ApplyMergedDelta(_xStar, _mergedDelta, _mergedCount, _mergedTouched, _mergedTouchedCount, omega, parallel, _opt);
                    _period.applyTetMs += _sw.Elapsed.TotalMilliseconds - tApply;
                }

                if (groundEnabled)
                {
                    double t0 = _sw.Elapsed.TotalMilliseconds;
                    PBDKernels.ProjectGroundFriction(
                        _x, _xStar, _w,
                        groundY, groundFrictionEnabled, groundMuStatic, groundMuDynamic, groundContactEps, groundSlop,
                        parallel, _opt);
                    _period.groundMs += _sw.Elapsed.TotalMilliseconds - t0;
                }
            }

            {
                double t0 = _sw.Elapsed.TotalMilliseconds;
                PBDKernels.Commit(_x, _v, _w, _xStar, invDt, groundEnabled, groundY, groundContactEps, groundTangentDamping, parallel, _opt);
                _period.commitMs += _sw.Elapsed.TotalMilliseconds - t0;
            }

            {
                double t0 = _sw.Elapsed.TotalMilliseconds;
                UploadAllBodies();
                _period.uploadMs += _sw.Elapsed.TotalMilliseconds - t0;
            }
        }

        _sw.Stop();
        _period.totalMs += _sw.Elapsed.TotalMilliseconds;
    }

    private static int SnapshotAccums(
        List<PBDKernels.LocalAccum> list, object gate,
        ref PBDKernels.LocalAccum[] snapshot)
    {
        lock (gate)
        {
            int c = list.Count;
            if (snapshot.Length < c) Array.Resize(ref snapshot, Math.Max(c, snapshot.Length * 2));
            list.CopyTo(0, snapshot, 0, c);
            return c;
        }
    }

    private void EnsureTLS()
    {
        if (!UseParallel || _x == null) return;
        if (_tlsEdge != null && _tlsTet != null) return;

        int threads = EffectiveThreadCount();
        int n = _x.Length;
        int edgeCount = _edgeI0 != null ? _edgeI0.Length : 0;
        int tetCount = _tetA != null ? _tetA.Length : 0;

        int edgeTouchedCap = PBDKernels.EstimateTouchedCapacityEdges(n, edgeCount, threads);
        int tetTouchedCap = PBDKernels.EstimateTouchedCapacityTets(n, tetCount, threads);

        DisposeTLS();

        lock (_edgeRegLock) _edgeRegistered.Clear();
        lock (_tetRegLock) _tetRegistered.Clear();

        _tlsEdge = new ThreadLocal<PBDKernels.LocalAccum>(() =>
        {
            var a = new PBDKernels.LocalAccum(n, edgeTouchedCap);
            lock (_edgeRegLock) _edgeRegistered.Add(a);
            return a;
        }, true);

        _tlsTet = new ThreadLocal<PBDKernels.LocalAccum>(() =>
        {
            var a = new PBDKernels.LocalAccum(n, tetTouchedCap);
            lock (_tetRegLock) _tetRegistered.Add(a);
            return a;
        }, true);
    }

    private void DisposeTLS()
    {
        _tlsEdge?.Dispose(); _tlsEdge = null;
        _tlsTet?.Dispose(); _tlsTet = null;
    }

    private void RebuildWorld()
    {
        _dirtyRebuild = false;

        int totalParticles = 0;
        int totalEdges = 0;
        int totalTets = 0;

        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b == null || b.runtimeMesh == null || !b.runtimeMesh.IsValid) continue;
            totalParticles += b.runtimeMesh.vertices.Length;
            totalEdges += b.runtimeMesh.edgeIds.Length / 2;
            totalTets += b.runtimeMesh.tetIds.Length / 4;
        }

        if (totalParticles <= 0)
        {
            _x = _xStar = _v = null;
            _w = null;
            DisposeTLS();
            return;
        }

        _x = new Vector3[totalParticles];
        _xStar = new Vector3[totalParticles];
        _v = new Vector3[totalParticles];
        _w = new float[totalParticles];

        _edgeI0 = new int[totalEdges];
        _edgeI1 = new int[totalEdges];
        _edgeRest = new float[totalEdges];
        _edgeCompliance = new float[totalEdges];
        _edgeLambda = new float[totalEdges];

        _tetA = new int[totalTets];
        _tetB = new int[totalTets];
        _tetC = new int[totalTets];
        _tetD = new int[totalTets];
        _tetRestVol = new float[totalTets];
        _tetCompliance = new float[totalTets];
        _tetLambda = new float[totalTets];

        _mergedDelta = new Vector3[totalParticles];
        _mergedCount = new int[totalParticles];
        _mergedTouched = new int[totalParticles];
        _mergedTouchedCount = 0;

        int pCursor = 0;
        int eCursor = 0;
        int tCursor = 0;

        for (int bi = 0; bi < _bodies.Count; bi++)
        {
            var body = _bodies[bi];
            if (body == null || body.runtimeMesh == null || !body.runtimeMesh.IsValid) continue;

            var rm = body.runtimeMesh;

            int pCount = rm.vertices.Length;
            int eCount = rm.edgeIds.Length / 2;
            int tCount = rm.tetIds.Length / 4;

            body.globalParticleStart = pCursor;
            body.globalParticleCount = pCount;
            body.globalEdgeStart = eCursor;
            body.globalEdgeCount = eCount;
            body.globalTetStart = tCursor;
            body.globalTetCount = tCount;

            body.GetInitialWorldPositions(_x, pCursor);
            Array.Copy(_x, pCursor, _xStar, pCursor, pCount);

            for (int i = 0; i < pCount; i++) _w[pCursor + i] = 0f;

            var tetIds = rm.tetIds;
            for (int ti = 0; ti < tCount; ti++)
            {
                int k = ti * 4;
                int a = pCursor + tetIds[k + 0];
                int b = pCursor + tetIds[k + 1];
                int c = pCursor + tetIds[k + 2];
                int d = pCursor + tetIds[k + 3];

                _tetA[tCursor + ti] = a;
                _tetB[tCursor + ti] = b;
                _tetC[tCursor + ti] = c;
                _tetD[tCursor + ti] = d;

                float vol = PBDKernels.TetVolume(_x[a], _x[b], _x[c], _x[d]);
                _tetRestVol[tCursor + ti] = vol;
                _tetCompliance[tCursor + ti] = Mathf.Max(0f, body.volumeCompliance);
                _tetLambda[tCursor + ti] = 0f;

                float mvol = Mathf.Abs(vol);
                if (mvol > 1e-12f)
                {
                    float inv = 4f / mvol;
                    _w[a] += inv; _w[b] += inv; _w[c] += inv; _w[d] += inv;
                }
            }

            body.ApplyPinning(_w, pCursor);

            var edgeIds = rm.edgeIds;
            for (int ei = 0; ei < eCount; ei++)
            {
                int u = pCursor + edgeIds[2 * ei + 0];
                int v = pCursor + edgeIds[2 * ei + 1];

                _edgeI0[eCursor + ei] = u;
                _edgeI1[eCursor + ei] = v;
                _edgeRest[eCursor + ei] = (_x[v] - _x[u]).magnitude;
                _edgeCompliance[eCursor + ei] = Mathf.Max(0f, body.edgeCompliance);
                _edgeLambda[eCursor + ei] = 0f;
            }

            pCursor += pCount;
            eCursor += eCount;
            tCursor += tCount;
        }

        DisposeTLS();
    }

    private void SolveEdgesSerial(float dt)
    {
        PBDKernels.ResetMerged(_mergedDelta, _mergedCount, _mergedTouched, ref _mergedTouchedCount);

        float invDt2 = dt > 1e-12f ? 1f / (dt * dt) : 0f;
        int m = _edgeI0.Length;

        for (int ei = 0; ei < m; ei++)
        {
            int i0 = _edgeI0[ei];
            int i1 = _edgeI1[ei];
            float w0 = _w[i0], w1 = _w[i1];
            float wSum = w0 + w1;
            if (wSum == 0f) continue;

            Vector3 p0 = _xStar[i0];
            Vector3 p1 = _xStar[i1];

            Vector3 d = p0 - p1;
            float len = d.magnitude;
            if (len < 1e-12f) continue;

            float C = len - _edgeRest[ei];

            float alpha = _edgeCompliance[ei] * invDt2;
            float lambda = _edgeLambda[ei];
            float dlambda = (-C - alpha * lambda) / (wSum + alpha);
            _edgeLambda[ei] = lambda + dlambda;

            Vector3 corr = (dlambda / len) * d;

            AddMerged(i0, w0 * corr);
            AddMerged(i1, -w1 * corr);
        }
    }

    private void SolveTetsSerial(float dt)
    {
        PBDKernels.ResetMerged(_mergedDelta, _mergedCount, _mergedTouched, ref _mergedTouchedCount);

        float invDt2 = dt > 1e-12f ? 1f / (dt * dt) : 0f;
        int m = _tetA.Length;

        for (int ti = 0; ti < m; ti++)
        {
            int a = _tetA[ti], b = _tetB[ti], c = _tetC[ti], d = _tetD[ti];
            float wa = _w[a], wb = _w[b], wc = _w[c], wd = _w[d];
            if (wa + wb + wc + wd == 0f) continue;

            Vector3 pa = _xStar[a];
            Vector3 pb = _xStar[b];
            Vector3 pc = _xStar[c];
            Vector3 pd = _xStar[d];

            Vector3 ga = Vector3.Cross(pd - pb, pc - pb) / 6f;
            Vector3 gb = Vector3.Cross(pc - pa, pd - pa) / 6f;
            Vector3 gc = Vector3.Cross(pd - pa, pb - pa) / 6f;
            Vector3 gd = Vector3.Cross(pb - pa, pc - pa) / 6f;

            float wSum =
                wa * Vector3.Dot(ga, ga) +
                wb * Vector3.Dot(gb, gb) +
                wc * Vector3.Dot(gc, gc) +
                wd * Vector3.Dot(gd, gd);

            if (wSum < 1e-20f) continue;

            float vol = PBDKernels.TetVolume(pa, pb, pc, pd);
            float C = vol - _tetRestVol[ti];

            float alpha = _tetCompliance[ti] * invDt2;
            float lambda = _tetLambda[ti];
            float dlambda = (-C - alpha * lambda) / (wSum + alpha);
            _tetLambda[ti] = lambda + dlambda;

            AddMerged(a, wa * (dlambda * ga));
            AddMerged(b, wb * (dlambda * gb));
            AddMerged(c, wc * (dlambda * gc));
            AddMerged(d, wd * (dlambda * gd));
        }
    }

    private void AddMerged(int i, Vector3 d)
    {
        if (_mergedCount[i] == 0) _mergedTouched[_mergedTouchedCount++] = i;
        _mergedDelta[i] += d;
        _mergedCount[i] += 1;
    }

    private void UploadAllBodies()
    {
        for (int i = 0; i < _bodies.Count; i++)
        {
            var b = _bodies[i];
            if (b != null) b.UploadFromWorld(_x);
        }
    }

    private void ApplyThreadCount()
    {
        int tc = EffectiveThreadCount();
        _opt.MaxDegreeOfParallelism = tc;
    }

    private void TickStats()
    {
        if (!printStats) return;

        _statsFrames++;
        _statsTimer += Time.unscaledDeltaTime;

        float period = Mathf.Max(0.1f, statsPeriodSeconds);
        if (_statsTimer < period) return;

        float fps = _statsFrames / _statsTimer;

        double total = Math.Max(1e-9, _period.totalMs);
        double pct(double x) => 100.0 * x / total;

        double eTotal = _period.edgeComputeMs + _period.edgeMergeMs + _period.applyEdgeMs;
        double tTotal = _period.tetComputeMs + _period.tetMergeMs + _period.applyTetMs;

        UnityEngine.Debug.Log(
            $"[PBDWorld] Mode {executionMode} Threads {EffectiveThreadCount()} FPS {fps:F1} Bodies {_bodies.Count} " +
            $"Total {total:F2}ms | " +
            $"Ecomp {_period.edgeComputeMs:F2}({pct(_period.edgeComputeMs):F1}%) " +
            $"Emrg {_period.edgeMergeMs:F2}({pct(_period.edgeMergeMs):F1}%) " +
            $"Eapp {_period.applyEdgeMs:F2}({pct(_period.applyEdgeMs):F1}%) " +
            $"E {eTotal:F2}({pct(eTotal):F1}%) | " +
            $"Tcomp {_period.tetComputeMs:F2}({pct(_period.tetComputeMs):F1}%) " +
            $"Tmrg {_period.tetMergeMs:F2}({pct(_period.tetMergeMs):F1}%) " +
            $"Tapp {_period.applyTetMs:F2}({pct(_period.applyTetMs):F1}%) " +
            $"T {tTotal:F2}({pct(tTotal):F1}%) | " +
            $"Rebuild {_period.rebuildMs:F2}({pct(_period.rebuildMs):F1}%) " +
            $"Pred {_period.predictMs:F2}({pct(_period.predictMs):F1}%) " +
            $"G {_period.groundMs:F2}({pct(_period.groundMs):F1}%) " +
            $"C {_period.commitMs:F2}({pct(_period.commitMs):F1}%) " +
            $"U {_period.uploadMs:F2}({pct(_period.uploadMs):F1}%)"
        );

        _statsFrames = 0;
        _statsTimer = 0f;
        _period.Clear();
    }
}
