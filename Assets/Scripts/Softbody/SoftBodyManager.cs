// SoftBodyManager.cs
// 请以 UTF-8 保存本文件
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

public sealed class SoftBodyManager : MonoBehaviour
{
    public enum ComputeMode
    {
        SingleThread = 0,
        MultiThreadWithinSolver = 1
    }

    private struct FrameTimers
    {
        public double TotalMs, CacheMs, PreMs, SolveMs, CollidersMs, PostMs, UploadMs, NormalsMs;
        public void Clear() => TotalMs = CacheMs = PreMs = SolveMs = CollidersMs = PostMs = UploadMs = NormalsMs = 0.0;
    }

    public static SoftBodyManager Instance { get; private set; }

    [Header("Simulation")]
    [SerializeField] private Vector3 gravity = new Vector3(0f, -10f, 0f);
    [SerializeField, Min(1)] private int substeps = 8;
    [SerializeField, Min(1)] private int solverIterations = 6;

    [SerializeField] private bool simulateInFixedUpdate = false;
    [SerializeField] private float fixedDtOverride = 1f / 60f;

    [Header("Update Fixed-Timestep (recommended)")]
    [SerializeField, Min(1)] private int maxStepsPerFrame = 4;
    [SerializeField, Min(0.001f)] private float maxFrameDeltaTime = 0.05f;

    [Header("Threading")]
    [SerializeField] private ComputeMode computeMode = ComputeMode.SingleThread;
    [SerializeField] private int maxWorkerThreads = 0;

    [Header("Auto Registration")]
    [SerializeField] private bool autoRegisterAllSolversOnEnable = true;
    [SerializeField] private bool autoRegisterAllPrimitiveCollidersOnEnable = true;

    [Header("Stats")]
    [SerializeField] private bool printStats = true;
    [SerializeField] private float statsPeriodSeconds = 1f;

    private readonly List<SoftBodySolver> solvers = new List<SoftBodySolver>(64);
    private readonly List<SoftBodyPrimitiveCollider> primitiveColliders = new List<SoftBodyPrimitiveCollider>(64);

    private SoftBodyPrimitiveCollider.PrimitiveColliderData[] colliderCache = new SoftBodyPrimitiveCollider.PrimitiveColliderData[0];
    private int colliderCacheCount = 0;

    private readonly Stopwatch swSeg = new Stopwatch();
    private readonly Stopwatch swTotal = new Stopwatch();

    private float fpsTimer;
    private int fpsFrames;
    private FrameTimers period;

    private float accumulator;

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    private void OnEnable()
    {
        if (autoRegisterAllSolversOnEnable) RegisterAllSolversInScene();
        if (autoRegisterAllPrimitiveCollidersOnEnable) RegisterAllPrimitiveCollidersInScene();
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }

    private float GetFixedStepDt()
    {
        return fixedDtOverride > 0f ? fixedDtOverride : Time.fixedDeltaTime;
    }

    private void Update()
    {
        if (!simulateInFixedUpdate)
        {
            float frameDt = Time.deltaTime;
            if (frameDt > maxFrameDeltaTime) frameDt = maxFrameDeltaTime;
            accumulator += frameDt;

            float stepDt = GetFixedStepDt();

            int steps = 0;
            while (accumulator >= stepDt && steps < maxStepsPerFrame)
            {
                StepSimulation(stepDt);
                accumulator -= stepDt;
                steps++;
            }

            if (steps >= maxStepsPerFrame)
                accumulator = 0f;
        }

        TickStats();
    }

    private void FixedUpdate()
    {
        if (!simulateInFixedUpdate) return;
        StepSimulation(GetFixedStepDt());
    }

    private void LateUpdate()
    {
        if (solvers.Count == 0) return;

        swTotal.Restart();

        swSeg.Restart();
        for (int i = 0; i < solvers.Count; i++) solvers[i].UploadMeshVerticesBoundsMainThread();
        swSeg.Stop();
        period.UploadMs += swSeg.Elapsed.TotalMilliseconds;

        swSeg.Restart();
        for (int i = 0; i < solvers.Count; i++) solvers[i].UploadMeshNormalsMainThread();
        swSeg.Stop();
        period.NormalsMs += swSeg.Elapsed.TotalMilliseconds;

        swTotal.Stop();
        period.TotalMs += swTotal.Elapsed.TotalMilliseconds;
    }

    // ---------- Public API ----------
    public void Register(SoftBodySolver solver)
    {
        if (solver == null) return;
        if (!solvers.Contains(solver)) solvers.Add(solver);
    }

    public void Unregister(SoftBodySolver solver)
    {
        if (solver == null) return;
        solvers.Remove(solver);
    }

    public void RegisterAllSolversInScene()
    {
        var all = FindObjectsOfType<SoftBodySolver>(true);
        for (int i = 0; i < all.Length; i++) Register(all[i]);
    }

    public void RegisterPrimitiveCollider(SoftBodyPrimitiveCollider c)
    {
        if (c == null) return;
        if (!primitiveColliders.Contains(c)) primitiveColliders.Add(c);
    }

    public void UnregisterPrimitiveCollider(SoftBodyPrimitiveCollider c)
    {
        if (c == null) return;
        primitiveColliders.Remove(c);
    }

    public void RegisterAllPrimitiveCollidersInScene()
    {
        var all = FindObjectsOfType<SoftBodyPrimitiveCollider>(true);
        for (int i = 0; i < all.Length; i++) RegisterPrimitiveCollider(all[i]);
    }

    // ---------- Stats ----------
    private void TickStats()
    {
        if (!printStats) return;

        fpsFrames++;
        fpsTimer += Time.unscaledDeltaTime;

        float p = statsPeriodSeconds > 0.05f ? statsPeriodSeconds : 1f;
        if (fpsTimer < p) return;

        float fps = fpsFrames / fpsTimer;

        double denom = period.TotalMs > 1e-9 ? period.TotalMs : 1.0;
        double pcCache = 100.0 * period.CacheMs / denom;
        double pcPre = 100.0 * period.PreMs / denom;
        double pcSolve = 100.0 * period.SolveMs / denom;
        double pcCol = 100.0 * period.CollidersMs / denom;
        double pcPost = 100.0 * period.PostMs / denom;
        double pcUp = 100.0 * period.UploadMs / denom;
        double pcN = 100.0 * period.NormalsMs / denom;

        int threadCount = maxWorkerThreads > 0 ? maxWorkerThreads : System.Environment.ProcessorCount;
        string modeStr = computeMode == ComputeMode.SingleThread ? "ST" : "MT";

        UnityEngine.Debug.Log(
            $"FPS {fps:F1} Bodies {solvers.Count} Mode {modeStr} Threads {threadCount} " +
            $"TotalMs {period.TotalMs:F3} " +
            $"Cache {period.CacheMs:F3}({pcCache:F1}%) " +
            $"Pre {period.PreMs:F3}({pcPre:F1}%) " +
            $"Solve {period.SolveMs:F3}({pcSolve:F1}%) " +
            $"Coll {period.CollidersMs:F3}({pcCol:F1}%) " +
            $"Post {period.PostMs:F3}({pcPost:F1}%) " +
            $"Upload {period.UploadMs:F3}({pcUp:F1}%) " +
            $"Norm {period.NormalsMs:F3}({pcN:F1}%)"
        );

        fpsFrames = 0;
        fpsTimer = 0f;
        period.Clear();
    }

    // ---------- Simulation ----------
    private void StepSimulation(float dt)
    {
        if (solvers.Count == 0) return;

        bool useParallel = computeMode == ComputeMode.MultiThreadWithinSolver;
        int threads = maxWorkerThreads > 0 ? maxWorkerThreads : System.Environment.ProcessorCount;

        swTotal.Restart();

        int ss = Mathf.Max(1, substeps);
        float sdt = dt / ss;

        swSeg.Restart();
        RebuildColliderCache();
        for (int i = 0; i < solvers.Count; i++)
            solvers[i].CacheStepDataMainThread(gravity, useParallel, threads, colliderCache, colliderCacheCount, solverIterations);
        swSeg.Stop();
        period.CacheMs += swSeg.Elapsed.TotalMilliseconds;

        for (int step = 0; step < ss; step++)
        {
            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].PreSolveWorkerSafe(sdt);
            swSeg.Stop();
            period.PreMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].SolveWorkerSafe(sdt);
            swSeg.Stop();
            period.SolveMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].PostSolveWorkerSafe(sdt);
            swSeg.Stop();
            period.PostMs += swSeg.Elapsed.TotalMilliseconds;
        }

        swTotal.Stop();
        period.TotalMs += swTotal.Elapsed.TotalMilliseconds;
    }

    private void RebuildColliderCache()
    {
        int count = 0;
        for (int i = 0; i < primitiveColliders.Count; i++)
        {
            var c = primitiveColliders[i];
            if (c == null) continue;
            if (!c.IsActiveForSoftBody) continue;
            count++;
        }

        if (colliderCache == null || colliderCache.Length < count)
            colliderCache = new SoftBodyPrimitiveCollider.PrimitiveColliderData[Mathf.NextPowerOfTwo(Mathf.Max(1, count))];

        colliderCacheCount = 0;
        for (int i = 0; i < primitiveColliders.Count; i++)
        {
            var c = primitiveColliders[i];
            if (c == null) continue;
            if (!c.IsActiveForSoftBody) continue;
            colliderCache[colliderCacheCount++] = c.GetWorldData();
        }
    }
}
