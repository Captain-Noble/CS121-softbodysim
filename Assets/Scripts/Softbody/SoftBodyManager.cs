// SoftBodyManager.cs
// 请以 UTF-8 保存本文件
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using UnityEngine;

public sealed class SoftBodyManager : MonoBehaviour
{
    public enum ComputeMode
    {
        SingleThread = 0,
        MultiThreadWithinSolver = 1,
        GpuCompute = 2
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

    [Tooltip("仍保留该选项，但推荐关闭：我们在 Update 中用固定步长 accumulator 来跑，更稳定也更快。")]
    [SerializeField] private bool simulateInFixedUpdate = false;

    [Tooltip("固定步长（秒）。>0 则强制使用该值，否则使用 Time.fixedDeltaTime")]
    [SerializeField] private float fixedDtOverride = 1f / 60f;

    [Header("Update Fixed-Timestep (recommended)")]
    [SerializeField, Min(1)] private int maxStepsPerFrame = 4;
    [SerializeField, Min(0.001f)] private float maxFrameDeltaTime = 0.05f;

    [Header("Compute")]
    [SerializeField] private ComputeMode computeMode = ComputeMode.SingleThread;
    [Tooltip("CPU 多线程时：0 = use Environment.ProcessorCount")]
    [SerializeField] private int maxWorkerThreads = 0;

    [Header("SoftBody ↔ SoftBody Collisions")]
    [SerializeField] private bool enableInterSoftBodyCollision = true;
    [Tooltip("每个 substep 做几次互撞迭代（1~2 通常够用）")]
    [SerializeField, Range(1, 4)] private int interCollisionIterations = 1;
    [Tooltip("互撞修正强度（1 = 一步推开；更大更硬，但可能抖）")]
    [SerializeField, Range(0.5f, 2.0f)] private float interCollisionOmega = 1.0f;
    [Tooltip("网格 cellSize = 2*maxRadius*multiplier（越大越粗糙但更快）")]
    [SerializeField, Range(1.0f, 3.0f)] private float interCollisionCellMultiplier = 1.5f;

    [Header("Stats")]
    [SerializeField] private bool printStats = true;
    [SerializeField] private float statsPeriodSeconds = 1f;
    [SerializeField] private bool autoRegisterAllSolversOnEnable = true;
    [SerializeField] private bool autoRegisterAllPrimitiveCollidersOnEnable = true;

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

    // -------------------------
    // Inter-softbody collision caches (reused to avoid GC)
    // -------------------------
    private int interTotal;
    private Vector3[] interPosW;
    private Vector3[] interDelta;
    private int[] interCount;
    private float[] interInvMass;
    private float[] interRadius;
    private int[] interSolverId;
    private int[] interLocalId;

    // spatial hash table (open addressing)
    private const long KEY_EMPTY = long.MinValue;
    private long[] hashKeys;
    private int[] hashHeads;
    private int[] hashNext;
    private int hashMask;

    private readonly ParallelOptions interOpt = new ParallelOptions();

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

    private float GetFixedStepDt() => fixedDtOverride > 0f ? fixedDtOverride : Time.fixedDeltaTime;

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

            if (steps >= maxStepsPerFrame) accumulator = 0f;
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
        string modeStr = computeMode == ComputeMode.SingleThread ? "ST"
                      : computeMode == ComputeMode.MultiThreadWithinSolver ? "MT"
                      : "GPU";

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

        int threads = maxWorkerThreads > 0 ? maxWorkerThreads : System.Environment.ProcessorCount;
        interOpt.MaxDegreeOfParallelism = threads;

        swTotal.Restart();

        int ss = Mathf.Max(1, substeps);
        float sdt = dt / ss;

        // Cache once per simulation step
        swSeg.Restart();
        RebuildColliderCache();
        for (int i = 0; i < solvers.Count; i++)
            solvers[i].CacheStepDataMainThread(gravity, computeMode, threads, colliderCache, colliderCacheCount, solverIterations);
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

            // ✅ SoftBody ↔ SoftBody collision pass (after each solver finished solve, before PostSolve)
            if (enableInterSoftBodyCollision && solvers.Count >= 2)
            {
                swSeg.Restart();
                SolveInterSoftBodyCollisions();
                swSeg.Stop();
                period.CollidersMs += swSeg.Elapsed.TotalMilliseconds;
            }

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

    // -------------------------
    // Inter-softbody collision
    // -------------------------
    private void EnsureInterCapacity(int totalParticles)
    {
        if (interPosW == null || interPosW.Length < totalParticles)
        {
            int cap = Mathf.NextPowerOfTwo(Mathf.Max(256, totalParticles));
            interPosW = new Vector3[cap];
            interDelta = new Vector3[cap];
            interCount = new int[cap];
            interInvMass = new float[cap];
            interRadius = new float[cap];
            interSolverId = new int[cap];
            interLocalId = new int[cap];
            hashNext = new int[cap];
        }

        // hash table size: ~2x particles
        int tableSize = Mathf.NextPowerOfTwo(Mathf.Max(512, totalParticles * 2));
        if (hashKeys == null || hashKeys.Length != tableSize)
        {
            hashKeys = new long[tableSize];
            hashHeads = new int[tableSize];
            hashMask = tableSize - 1;
        }

        // clear hash
        Array.Fill(hashKeys, KEY_EMPTY);
        Array.Fill(hashHeads, -1);
    }

    // pack 3 signed 21-bit coords into 63-bit key
    private const int CELL_BIAS = 1 << 20;        // 1,048,576
    private const int CELL_MASK = (1 << 21) - 1;  // 2,097,151

    private static long PackCellKey(int cx, int cy, int cz)
    {
        // clamp to safe range
        cx = Mathf.Clamp(cx, -CELL_BIAS, CELL_BIAS - 1);
        cy = Mathf.Clamp(cy, -CELL_BIAS, CELL_BIAS - 1);
        cz = Mathf.Clamp(cz, -CELL_BIAS, CELL_BIAS - 1);

        long x = (long)(cx + CELL_BIAS) & CELL_MASK;
        long y = (long)(cy + CELL_BIAS) & CELL_MASK;
        long z = (long)(cz + CELL_BIAS) & CELL_MASK;
        return (x << 42) | (y << 21) | z;
    }

    private int HashFindSlot(long key)
    {
        // 64-bit mix -> slot
        unchecked
        {
            ulong x = (ulong)key;
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdUL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53UL;
            x ^= x >> 33;
            int slot = (int)x & hashMask;

            while (true)
            {
                long k = hashKeys[slot];
                if (k == KEY_EMPTY || k == key) return slot;
                slot = (slot + 1) & hashMask;
            }
        }
    }

    private int HashGetHead(long key)
    {
        int slot = HashFindSlot(key);
        if (hashKeys[slot] != key) return -1;
        return hashHeads[slot];
    }

    private void HashInsert(int particleIndex, long key)
    {
        int slot = HashFindSlot(key);
        if (hashKeys[slot] == KEY_EMPTY) hashKeys[slot] = key;

        hashNext[particleIndex] = hashHeads[slot];
        hashHeads[slot] = particleIndex;
    }

    private void SolveInterSoftBodyCollisions()
    {
        // 1) collect all particles from enabled solvers
        interTotal = 0;
        float maxR = 0f;

        for (int s = 0; s < solvers.Count; s++)
        {
            var solver = solvers[s];
            if (solver == null || !solver.isActiveAndEnabled) continue;
            if (!solver.InterSoftBodyCollisionEnabled) continue;

            // GPU solver: download predicted positions if needed (only when inter-collision enabled)
            solver.DownloadGpuPosPredToCpuIfNeeded();

            int n = solver.ParticleCount;
            float r = solver.ParticleRadius;
            maxR = Mathf.Max(maxR, r);

            EnsureInterCapacity(interTotal + n);

            var posPredL = solver.PosPredArrayCpu;
            var invMass = solver.InvMassArray;

            Matrix4x4 l2w = solver.CachedLocalToWorld;

            for (int i = 0; i < n; i++)
            {
                int idx = interTotal++;
                interSolverId[idx] = s;
                interLocalId[idx] = i;
                interInvMass[idx] = invMass[i];
                interRadius[idx] = r;
                interPosW[idx] = l2w.MultiplyPoint3x4(posPredL[i]);
            }
        }

        if (interTotal <= 0) return;

        float cellSize = Mathf.Max(1e-4f, 2f * maxR * interCollisionCellMultiplier);
        float invCell = 1f / cellSize;

        // 2) iterative collision relax (usually 1 is enough)
        int iters = Mathf.Clamp(interCollisionIterations, 1, 4);

        bool allowParallel = (computeMode != ComputeMode.SingleThread);
        for (int iter = 0; iter < iters; iter++)
        {
            // rebuild hash each iteration (positions changed)
            Array.Fill(hashKeys, KEY_EMPTY);
            Array.Fill(hashHeads, -1);

            for (int i = 0; i < interTotal; i++)
            {
                Vector3 p = interPosW[i];
                int cx = Mathf.FloorToInt(p.x * invCell);
                int cy = Mathf.FloorToInt(p.y * invCell);
                int cz = Mathf.FloorToInt(p.z * invCell);
                long key = PackCellKey(cx, cy, cz);
                HashInsert(i, key);
            }

            // clear delta/count
            Array.Clear(interDelta, 0, interTotal);
            Array.Clear(interCount, 0, interTotal);

            Action<int> gatherOne = (i) =>
            {
                float wi = interInvMass[i];
                if (wi == 0f) return;

                Vector3 pi = interPosW[i];
                float ri = interRadius[i];

                int sI = interSolverId[i];

                int cx = Mathf.FloorToInt(pi.x * invCell);
                int cy = Mathf.FloorToInt(pi.y * invCell);
                int cz = Mathf.FloorToInt(pi.z * invCell);

                Vector3 sum = Vector3.zero;
                int cnt = 0;

                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            long key = PackCellKey(cx + dx, cy + dy, cz + dz);
                            int head = HashGetHead(key);
                            for (int j = head; j != -1; j = hashNext[j])
                            {
                                if (j == i) continue;
                                if (interSolverId[j] == sI) continue; // only inter-body

                                Vector3 pj = interPosW[j];
                                Vector3 d = pi - pj;
                                float dist2 = d.sqrMagnitude;

                                float rj = interRadius[j];
                                float R = ri + rj;
                                if (dist2 >= R * R) continue;

                                float wj = interInvMass[j];
                                float wsum = wi + wj;
                                if (wsum <= 0f) continue;

                                float dist = Mathf.Sqrt(Mathf.Max(dist2, 1e-18f));
                                Vector3 nrm = dist > 1e-9f ? (d / dist) : Vector3.up;

                                float overlap = R - dist;

                                // Jacobi style: each particle computes its own share
                                float share = wi / wsum;
                                sum += nrm * (overlap * share);
                                cnt++;
                            }
                        }

                interDelta[i] = sum;
                interCount[i] = cnt;
            };

            if (!allowParallel)
            {
                for (int i = 0; i < interTotal; i++) gatherOne(i);
            }
            else
            {
                Parallel.For(0, interTotal, interOpt, gatherOne);
            }

            // apply
            float omega = interCollisionOmega;
            for (int i = 0; i < interTotal; i++)
            {
                int cnt = interCount[i];
                if (cnt <= 0) continue;
                if (interInvMass[i] == 0f) continue;

                interPosW[i] += (omega / cnt) * interDelta[i];
            }
        }

        // 3) write back to each solver posPred (local)
        // First, set CPU posPred arrays
        for (int k = 0; k < interTotal; k++)
        {
            int s = interSolverId[k];
            int i = interLocalId[k];

            var solver = solvers[s];
            if (solver == null) continue;

            Matrix4x4 w2l = solver.CachedWorldToLocal;
            Vector3 pl = w2l.MultiplyPoint3x4(interPosW[k]);
            solver.PosPredArrayCpu[i] = pl;
        }

        // Then, for GPU solvers, upload corrected posPred back to GPU before PostSolve
        for (int s = 0; s < solvers.Count; s++)
        {
            var solver = solvers[s];
            if (solver == null || !solver.isActiveAndEnabled) continue;
            if (!solver.InterSoftBodyCollisionEnabled) continue;

            solver.UploadCpuPosPredToGpuIfNeeded();
        }
    }
}
