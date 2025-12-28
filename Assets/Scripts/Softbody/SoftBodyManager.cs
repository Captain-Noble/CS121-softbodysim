// SoftBodyManager.cs
// 请以 UTF-8 保存本文件
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

public sealed class SoftBodyManager : MonoBehaviour
{
    #region Types
    public enum ComputeMode
    {
        SingleThread = 0,
        MultiThreadWithinSolver = 1
    }

    private struct FrameTimers
    {
        public double TotalMs;
        public double CacheMs;
        public double PreMs;
        public double SolveMs;
        public double CollidersMs;
        public double PostMs;
        public double UploadMs;
        public double NormalsMs;

        public void Clear()
        {
            TotalMs = 0.0;
            CacheMs = 0.0;
            PreMs = 0.0;
            SolveMs = 0.0;
            CollidersMs = 0.0;
            PostMs = 0.0;
            UploadMs = 0.0;
            NormalsMs = 0.0;
        }
    }
    #endregion

    #region Singleton
    public static SoftBodyManager Instance { get; private set; }
    #endregion

    #region Inspector
    [SerializeField] private Vector3 gravity = new Vector3(0f, -10f, 0f);
    [SerializeField] private int substeps = 10;
    [SerializeField] private int solverIterations = 1;
    [SerializeField] private bool simulateInFixedUpdate = true;
    [SerializeField] private float fixedDtOverride = 1f / 60f;

    [SerializeField] private ComputeMode computeMode = ComputeMode.SingleThread;
    [SerializeField] private int maxWorkerThreads = 0;

    [SerializeField] private bool printStats = true;
    [SerializeField] private float statsPeriodSeconds = 1f;
    [SerializeField] private bool autoRegisterAllSolversOnEnable = true;
    #endregion

    #region State
    private readonly List<SoftBodySolver> solvers = new List<SoftBodySolver>(64);

    private readonly Stopwatch swTotal = new Stopwatch();
    private readonly Stopwatch swSeg = new Stopwatch();

    private float fpsTimer;
    private int fpsFrames;

    private FrameTimers period;
    #endregion

    #region Unity
    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    private void OnEnable()
    {
        if (autoRegisterAllSolversOnEnable) RegisterAllSolversInScene();
    }

    private void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }

    private void Update()
    {
        if (!simulateInFixedUpdate) StepAll(Time.deltaTime);
        TickStats();
    }

    private void FixedUpdate()
    {
        if (!simulateInFixedUpdate) return;
        float dt = fixedDtOverride > 0f ? fixedDtOverride : Time.fixedDeltaTime;
        StepAll(dt);
    }
    #endregion

    #region API
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
        var all = FindObjectsByType<SoftBodySolver>(FindObjectsSortMode.None);
        for (int i = 0; i < all.Length; i++) Register(all[i]);
    }

    public Vector3 Gravity => gravity;
    public int Substeps => Mathf.Max(1, substeps);
    public int SolverIterations => Mathf.Max(1, solverIterations);
    #endregion

    #region Stats
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
    #endregion

    #region Simulation
    private void StepAll(float dt)
    {
        if (solvers.Count == 0) return;

        bool mt = computeMode == ComputeMode.MultiThreadWithinSolver;
        int threads = maxWorkerThreads > 0 ? maxWorkerThreads : System.Environment.ProcessorCount;

        swTotal.Restart();

        int ss = Substeps;
        float sdt = dt / ss;

        for (int step = 0; step < ss; step++)
        {
            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].CacheStepDataMainThread(gravity, mt, threads);
            swSeg.Stop();
            period.CacheMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].PreSolveWorkerSafe(sdt);
            swSeg.Stop();
            period.PreMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int it = 0; it < SolverIterations; it++)
                for (int i = 0; i < solvers.Count; i++) solvers[i].SolveWorkerSafe(sdt);
            swSeg.Stop();
            period.SolveMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].SolveCollidersMainThread();
            swSeg.Stop();
            period.CollidersMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].PostSolveWorkerSafe(sdt);
            swSeg.Stop();
            period.PostMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].UploadMeshVerticesBoundsMainThread();
            swSeg.Stop();
            period.UploadMs += swSeg.Elapsed.TotalMilliseconds;

            swSeg.Restart();
            for (int i = 0; i < solvers.Count; i++) solvers[i].UploadMeshNormalsMainThread();
            swSeg.Stop();
            period.NormalsMs += swSeg.Elapsed.TotalMilliseconds;
        }

        swTotal.Stop();
        period.TotalMs += swTotal.Elapsed.TotalMilliseconds;
    }
    #endregion
}
