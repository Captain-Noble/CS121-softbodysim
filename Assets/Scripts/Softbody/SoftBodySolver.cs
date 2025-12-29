// SoftBodySolver.cs
// 请以 UTF-8 保存本文件
using Hanzzz.Tetrahedralizer;
using System;
using System.Threading.Tasks;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public sealed class SoftBodySolver : MonoBehaviour
{
    public enum ThreadingMode
    {
        InheritManager = 0,
        ForceSingleThread = 1,
        ForceMultithread = 2
    }

    [Header("Registration")]
    [SerializeField] private bool autoRegisterToManager = true;
    [SerializeField] private ThreadingMode threading = ThreadingMode.InheritManager;

    [Header("Mesh / Asset")]
    [SerializeField] private bool autoGenerateAssetFromMesh = true;
    [SerializeField] private MeshFilter sourceMeshFilter;
    [SerializeField] private SoftBodyTetMeshAsset meshAsset;
    [SerializeField][Range(0f, 1f)] private float degenerateTetrahedronRatio = 0f;

    [Header("Constraints (stiffness, NOT compliance)")]
    [Tooltip("0=完全不约束, 1=尽量刚性(需要足够 iterations/substeps)")]
    [SerializeField][Range(0f, 1f)] private float edgeStiffness = 0.9f;
    [Tooltip("体积保持强烈建议 >= 0.95")]
    [SerializeField][Range(0f, 1f)] private float volumeStiffness = 0.98f;

    [Header("Parallel Jacobi (constraint averaging + SOR)")]
    [SerializeField][Range(0.5f, 2.0f)] private float sorOmega = 1.4f;

    [Header("Mass")]
    [Tooltip("密度只影响惯性，不影响“能不能保持体积”。")]
    [SerializeField] private float density = 1f;

    [Header("Pinning")]
    [SerializeField] private bool pinTopLayer = false;
    [SerializeField] private float pinTopEpsilon = 1e-4f;

    [Header("Collisions")]
    [SerializeField] private bool primitiveCollision = true;
    [SerializeField] private float particleRadius = 0.02f;

    [Header("Ground (Plane)")]
    [Tooltip("打开后：粒子会被投影到地面平面之上（线程安全、便宜）")]
    [SerializeField] private bool collideWithGround = true;

    [Tooltip("如果设置了 Transform，则使用它的位置与 up 作为地面平面；否则使用下方 groundHeight/groundNormal")]
    [SerializeField] private Transform groundTransform;

    [Tooltip("当 groundTransform 为空时生效：地面高度（世界坐标）")]
    [SerializeField] private float groundHeight = 0f;

    [Tooltip("当 groundTransform 为空时生效：地面法线（世界坐标，自动归一化）")]
    [SerializeField] private Vector3 groundNormal = Vector3.up;

    [Header("Rendering")]
    [SerializeField] private bool updateNormals = true;

    // --------- runtime state ----------
    private int n;

    private Vector3[] posL;
    private Vector3[] posPrevL;
    private Vector3[] posPredL;
    private Vector3[] velL;
    private float[] invMass;

    private Vector3[] restPosL;

    private int[] edgeIds;
    private float[] restEdgeLen;

    private int[] tetIds;
    private float[] restTetVol;

    private int[] surfaceTris;

    // adjacency for gather
    private int[] edgesAdjOffsets, edgesAdjOther, edgesAdjEdge;
    private int[] tetsAdjOffsets, tetsAdjTet;
    private byte[] tetsAdjRole;

    // scratch
    private Vector3[] scratchDelta;
    private int[] scratchCount;

    // mesh
    private Mesh mesh;
    private Vector3[] renderVerts;

    // step cache
    private Matrix4x4 cachedLocalToWorld;
    private Matrix4x4 cachedWorldToLocal;
    private Vector3 cachedGravityL;

    private SoftBodyPrimitiveCollider.PrimitiveColliderData[] cachedColliders;
    private int cachedColliderCount;

    // ground cache (world-space plane)
    private bool cachedGroundEnabled;
    private Vector3 cachedGroundPointW;
    private Vector3 cachedGroundNormalW;

    private bool useParallel;
    private int cachedSolverIterations;
    private readonly ParallelOptions opt = new ParallelOptions();

    private void OnEnable()
    {
        EnsureAsset();
        BuildFromAsset();
        TryRegister();
    }

    private void OnDisable()
    {
        if (!autoRegisterToManager) return;
        if (SoftBodyManager.Instance != null) SoftBodyManager.Instance.Unregister(this);
    }

    private void TryRegister()
    {
        if (!autoRegisterToManager) return;

        var mgr = SoftBodyManager.Instance;
        if (mgr != null) { mgr.Register(this); return; }

        mgr = FindObjectOfType<SoftBodyManager>(true);
        if (mgr != null) mgr.Register(this);
    }

    // ---------- Main-thread cache ----------
    public void CacheStepDataMainThread(
        Vector3 gravityW,
        bool managerWantsParallel,
        int threads,
        SoftBodyPrimitiveCollider.PrimitiveColliderData[] colliderSnapshot,
        int colliderCount,
        int solverIterations)
    {
        cachedLocalToWorld = transform.localToWorldMatrix;
        cachedWorldToLocal = transform.worldToLocalMatrix;
        cachedGravityL = cachedWorldToLocal.MultiplyVector(gravityW);

        cachedColliders = colliderSnapshot;
        cachedColliderCount = colliderCount;

        cachedSolverIterations = Mathf.Max(1, solverIterations);

        useParallel = ResolveUseParallel(managerWantsParallel);
        opt.MaxDegreeOfParallelism = threads > 0 ? threads : Environment.ProcessorCount;

        // ---- ground plane cache (world) ----
        cachedGroundEnabled = collideWithGround;
        if (cachedGroundEnabled)
        {
            if (groundTransform != null)
            {
                cachedGroundPointW = groundTransform.position;
                cachedGroundNormalW = groundTransform.up;
            }
            else
            {
                cachedGroundPointW = new Vector3(0f, groundHeight, 0f);
                cachedGroundNormalW = groundNormal;
            }

            float n2 = cachedGroundNormalW.sqrMagnitude;
            if (n2 < 1e-12f) cachedGroundNormalW = Vector3.up;
            else cachedGroundNormalW /= Mathf.Sqrt(n2);
        }
    }

    private bool ResolveUseParallel(bool managerWantsParallel)
    {
        if (!isActiveAndEnabled) return false;
        if (threading == ThreadingMode.ForceSingleThread) return false;
        if (threading == ThreadingMode.ForceMultithread) return true;
        return managerWantsParallel;
    }

    // =======================
    // Worker-safe simulation
    // =======================
    public void PreSolveWorkerSafe(float dt)
    {
        // ST/MT 同一套 kernel，只差调度
        if (!useParallel) PreSolveSerial(dt);
        else Parallel.For(0, n, opt, i => PreSolveOne(i, dt));
    }

    public void SolveWorkerSafe(float dt)
    {
        int iters = cachedSolverIterations;

        for (int iter = 0; iter < iters; iter++)
        {
            // ST/MT: 동일 알고리즘 (gather -> apply)
            SolveEdgesGatherApply();
            SolveVolumesGatherApply();

            // collisions (thread-safe math only)
            if ((cachedGroundEnabled) || (primitiveCollision && cachedColliderCount > 0))
                SolveAllCollisions();
        }
    }

    public void PostSolveWorkerSafe(float dt)
    {
        if (!useParallel) PostSolveSerial(dt);
        else Parallel.For(0, n, opt, i => PostSolveOne(i, dt));

        Array.Copy(posL, renderVerts, n);
    }

    // ---------- Main-thread upload ----------
    public void UploadMeshVerticesBoundsMainThread()
    {
        if (mesh == null) return;
        mesh.vertices = renderVerts;
        mesh.RecalculateBounds();
    }

    public void UploadMeshNormalsMainThread()
    {
        if (!updateNormals) return;
        if (mesh == null) return;
        mesh.RecalculateNormals();
    }

    // =======================
    // Core steps
    // =======================
    private void PreSolveSerial(float dt)
    {
        for (int i = 0; i < n; i++) PreSolveOne(i, dt);
    }

    private void PreSolveOne(int i, float dt)
    {
        posPrevL[i] = posL[i];

        if (invMass[i] == 0f)
        {
            velL[i] = Vector3.zero;
            posPredL[i] = posL[i];
            return;
        }

        velL[i] += cachedGravityL * dt;
        posPredL[i] = posL[i] + velL[i] * dt;
    }

    private void PostSolveSerial(float dt)
    {
        for (int i = 0; i < n; i++) PostSolveOne(i, dt);
    }

    private void PostSolveOne(int i, float dt)
    {
        if (invMass[i] == 0f)
        {
            // pinned stays
            posL[i] = posPrevL[i];
            posPredL[i] = posPrevL[i];
            velL[i] = Vector3.zero;
            return;
        }

        float invDt = dt > 0f ? 1f / dt : 0f;
        velL[i] = (posPredL[i] - posPrevL[i]) * invDt;
        posL[i] = posPredL[i];
    }

    // =======================
    // Route A: particle-centric gather -> average -> apply
    // =======================
    private void SolveEdgesGatherApply()
    {
        if (edgeIds == null || restEdgeLen == null || restEdgeLen.Length == 0) return;

        if (!useParallel)
        {
            for (int i = 0; i < n; i++) GatherEdgesForParticle(i);
            for (int i = 0; i < n; i++) ApplyDelta(i);
        }
        else
        {
            Parallel.For(0, n, opt, i => GatherEdgesForParticle(i));
            Parallel.For(0, n, opt, i => ApplyDelta(i));
        }
    }

    private void GatherEdgesForParticle(int i)
    {
        if (invMass[i] == 0f) { scratchDelta[i] = Vector3.zero; scratchCount[i] = 0; return; }

        int begin = edgesAdjOffsets[i];
        int end = edgesAdjOffsets[i + 1];

        Vector3 xi = posPredL[i];
        float wi = invMass[i];

        Vector3 sum = Vector3.zero;
        int cnt = 0;

        for (int k = begin; k < end; k++)
        {
            int j = edgesAdjOther[k];
            float wj = invMass[j];
            float w = wi + wj;
            if (w == 0f) continue;

            int e = edgesAdjEdge[k];
            Vector3 xj = posPredL[j];

            Vector3 d = xi - xj;
            float len2 = d.sqrMagnitude;
            if (len2 < 1e-18f) continue;

            float len = Mathf.Sqrt(len2);
            float C = len - restEdgeLen[e];
            Vector3 nrm = d / len;

            // Jacobi gather: delta_i = -(wi/(wi+wj)) * C * n
            float lambda = -edgeStiffness * (C / w);
            sum += nrm * (lambda * wi);
            cnt++;
        }

        scratchDelta[i] = sum;
        scratchCount[i] = cnt;
    }

    private void SolveVolumesGatherApply()
    {
        if (tetIds == null || restTetVol == null || restTetVol.Length == 0) return;

        if (!useParallel)
        {
            for (int i = 0; i < n; i++) GatherVolumesForParticle(i);
            for (int i = 0; i < n; i++) ApplyDelta(i);
        }
        else
        {
            // ✅ 修复：只 gather 一次（你原来贴的代码这里 gather 了两次）
            Parallel.For(0, n, opt, i => GatherVolumesForParticle(i));
            Parallel.For(0, n, opt, i => ApplyDelta(i));
        }
    }

    private void GatherVolumesForParticle(int i)
    {
        if (invMass[i] == 0f) { scratchDelta[i] = Vector3.zero; scratchCount[i] = 0; return; }

        int begin = tetsAdjOffsets[i];
        int end = tetsAdjOffsets[i + 1];

        Vector3 sum = Vector3.zero;
        int cnt = 0;

        for (int kk = begin; kk < end; kk++)
        {
            int t = tetsAdjTet[kk];
            int role = tetsAdjRole[kk];

            int baseId = 4 * t;
            int a = tetIds[baseId + 0];
            int b = tetIds[baseId + 1];
            int c = tetIds[baseId + 2];
            int d = tetIds[baseId + 3];

            float wa = invMass[a];
            float wb = invMass[b];
            float wc = invMass[c];
            float wd = invMass[d];
            if (wa + wb + wc + wd == 0f) continue;

            Vector3 pa = posPredL[a];
            Vector3 pb = posPredL[b];
            Vector3 pc = posPredL[c];
            Vector3 pd = posPredL[d];

            // gradients of signed volume wrt each vertex
            Vector3 ga = Vector3.Cross(pd - pb, pc - pb) / 6f;
            Vector3 gb = Vector3.Cross(pc - pa, pd - pa) / 6f;
            Vector3 gc = Vector3.Cross(pd - pa, pb - pa) / 6f;
            Vector3 gd = Vector3.Cross(pb - pa, pc - pa) / 6f;

            float wsum =
                wa * Vector3.Dot(ga, ga) +
                wb * Vector3.Dot(gb, gb) +
                wc * Vector3.Dot(gc, gc) +
                wd * Vector3.Dot(gd, gd);

            if (wsum < 1e-20f) continue;

            float vol = TetSignedVolume(pa, pb, pc, pd);
            float C = vol - restTetVol[t];

            float lambda = -volumeStiffness * (C / wsum);

            Vector3 g;
            float wi;
            switch (role)
            {
                case 0: g = ga; wi = wa; break;
                case 1: g = gb; wi = wb; break;
                case 2: g = gc; wi = wc; break;
                default: g = gd; wi = wd; break;
            }

            if (wi == 0f) continue;

            sum += g * (lambda * wi);
            cnt++;
        }

        scratchDelta[i] = sum;
        scratchCount[i] = cnt;
    }

    private void ApplyDelta(int i)
    {
        int cnt = scratchCount[i];
        if (cnt <= 0) return;
        if (invMass[i] == 0f) return;

        // average + SOR
        posPredL[i] += (sorOmega / cnt) * scratchDelta[i];
    }

    // =======================
    // Collisions (Ground + Primitive)
    // =======================
    private void SolveAllCollisions()
    {
        if (!useParallel)
        {
            for (int i = 0; i < n; i++) SolveCollisionsForParticle(i);
        }
        else
        {
            Parallel.For(0, n, opt, i => SolveCollisionsForParticle(i));
        }
    }

    private void SolveCollisionsForParticle(int i)
    {
        if (invMass[i] == 0f) return;

        Vector3 pw = cachedLocalToWorld.MultiplyPoint3x4(posPredL[i]);
        float r = Mathf.Max(1e-6f, particleRadius);

        // 1) Ground plane (push out to the positive half-space)
        if (cachedGroundEnabled)
        {
            float dist = Vector3.Dot(cachedGroundNormalW, pw - cachedGroundPointW);
            if (dist < r)
            {
                pw += cachedGroundNormalW * (r - dist);
            }
        }

        // 2) Primitive colliders
        if (primitiveCollision && cachedColliderCount > 0)
        {
            for (int c = 0; c < cachedColliderCount; c++)
            {
                if (SoftBodyCollisionMath.ComputePushOut(cachedColliders[c], pw, r, out Vector3 pushW))
                    pw += pushW;
            }
        }

        posPredL[i] = cachedWorldToLocal.MultiplyPoint3x4(pw);
    }

    // =======================
    // Build / asset
    // =======================
    private void EnsureAsset()
    {
        if (meshAsset != null && meshAsset.IsValid()) return;

        if (!autoGenerateAssetFromMesh)
            throw new InvalidOperationException("SoftBodyTetMeshAsset missing");

        if (sourceMeshFilter == null) sourceMeshFilter = GetComponent<MeshFilter>();
        if (sourceMeshFilter == null) throw new InvalidOperationException("MeshFilter missing");

        Mesh src = sourceMeshFilter.sharedMesh;
        if (src == null) throw new InvalidOperationException("Source mesh missing");

        if (!SoftBodyTetMeshAsset.TryCreateFromMesh(src, degenerateTetrahedronRatio, out SoftBodyTetMeshAsset generated))
            throw new InvalidOperationException("Tetrahedralization failed");

        meshAsset = generated;
    }

    private void BuildFromAsset()
    {
        if (meshAsset == null || !meshAsset.IsValid())
            throw new InvalidOperationException("Invalid SoftBodyTetMeshAsset");

        restPosL = (Vector3[])meshAsset.vertices.Clone();
        n = restPosL.Length;

        posL = new Vector3[n];
        posPrevL = new Vector3[n];
        posPredL = new Vector3[n];
        velL = new Vector3[n];
        invMass = new float[n];

        Array.Copy(restPosL, posL, n);
        Array.Copy(restPosL, posPrevL, n);
        Array.Copy(restPosL, posPredL, n);

        tetIds = (int[])meshAsset.tetIds.Clone();
        int tetCount = tetIds.Length / 4;
        restTetVol = new float[tetCount];

        // ---- Correct mass build: accumulate mass, then invert ----
        // ✅ 质量必须用 |volume|（否则有负体积四面体会得到“接近 0 的质量”，invMass 爆炸导致坍缩/炸裂）
        float[] mass = new float[n];
        float dens = Mathf.Max(1e-6f, density);

        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            int a = tetIds[k + 0];
            int b = tetIds[k + 1];
            int c = tetIds[k + 2];
            int d = tetIds[k + 3];

            float signedVol = TetSignedVolume(restPosL[a], restPosL[b], restPosL[c], restPosL[d]);
            float absVol = Mathf.Abs(signedVol);

            float m = dens * Mathf.Max(1e-12f, absVol);
            float share = m * 0.25f;

            mass[a] += share;
            mass[b] += share;
            mass[c] += share;
            mass[d] += share;

            // 体积约束仍然用 signed volume（保持定向体积）
            restTetVol[t] = signedVol;
        }

        for (int i = 0; i < n; i++)
            invMass[i] = mass[i] > 0f ? 1f / mass[i] : 0f;

        // pin top
        if (pinTopLayer)
        {
            float topY = float.NegativeInfinity;
            for (int i = 0; i < n; i++) if (restPosL[i].y > topY) topY = restPosL[i].y;
            for (int i = 0; i < n; i++)
                if (Mathf.Abs(restPosL[i].y - topY) <= pinTopEpsilon) invMass[i] = 0f;
        }

        edgeIds = (int[])meshAsset.edgeIds.Clone();
        int edgeCount = edgeIds.Length / 2;
        restEdgeLen = new float[edgeCount];
        for (int e = 0; e < edgeCount; e++)
        {
            int a = edgeIds[2 * e + 0];
            int b = edgeIds[2 * e + 1];
            restEdgeLen[e] = (restPosL[b] - restPosL[a]).magnitude;
        }

        surfaceTris = (int[])meshAsset.surfaceTriIds.Clone();

        BuildEdgeAdjacency(n, edgeIds, out edgesAdjOffsets, out edgesAdjOther, out edgesAdjEdge);
        BuildTetAdjacency(n, tetIds, out tetsAdjOffsets, out tetsAdjTet, out tetsAdjRole);

        scratchDelta = new Vector3[n];
        scratchCount = new int[n];

        mesh = new Mesh();
        mesh.name = "SoftBodySurface";
        mesh.indexFormat = n > 65535 ? UnityEngine.Rendering.IndexFormat.UInt32 : UnityEngine.Rendering.IndexFormat.UInt16;
        mesh.MarkDynamic();

        renderVerts = new Vector3[n];
        Array.Copy(posL, renderVerts, n);

        mesh.vertices = renderVerts;
        mesh.triangles = surfaceTris;
        mesh.RecalculateBounds();
        if (updateNormals) mesh.RecalculateNormals();

        GetComponent<MeshFilter>().sharedMesh = mesh;
    }

    // =======================
    // Math
    // =======================
    private static float TetSignedVolume(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
    {
        Vector3 a = p1 - p0;
        Vector3 b = p2 - p0;
        Vector3 c = p3 - p0;
        return Vector3.Dot(Vector3.Cross(a, b), c) / 6f;
    }

    // =======================
    // Adjacency build
    // =======================
    private static void BuildEdgeAdjacency(
        int numParticles,
        int[] edgeIds,
        out int[] offsets,
        out int[] other,
        out int[] edgeIndex)
    {
        int edgeCount = edgeIds != null ? (edgeIds.Length / 2) : 0;
        offsets = new int[numParticles + 1];

        if (edgeCount == 0)
        {
            other = Array.Empty<int>();
            edgeIndex = Array.Empty<int>();
            return;
        }

        for (int e = 0; e < edgeCount; e++)
        {
            int a = edgeIds[2 * e + 0];
            int b = edgeIds[2 * e + 1];
            offsets[a + 1]++;
            offsets[b + 1]++;
        }

        for (int i = 0; i < numParticles; i++)
            offsets[i + 1] += offsets[i];

        int total = offsets[numParticles];
        other = new int[total];
        edgeIndex = new int[total];

        int[] cur = new int[numParticles];
        Array.Copy(offsets, cur, numParticles);

        for (int e = 0; e < edgeCount; e++)
        {
            int a = edgeIds[2 * e + 0];
            int b = edgeIds[2 * e + 1];

            int ka = cur[a]++;
            other[ka] = b;
            edgeIndex[ka] = e;

            int kb = cur[b]++;
            other[kb] = a;
            edgeIndex[kb] = e;
        }
    }

    private static void BuildTetAdjacency(
        int numParticles,
        int[] tetIds,
        out int[] offsets,
        out int[] tetIndex,
        out byte[] role)
    {
        int tetCount = tetIds != null ? (tetIds.Length / 4) : 0;
        offsets = new int[numParticles + 1];

        if (tetCount == 0)
        {
            tetIndex = Array.Empty<int>();
            role = Array.Empty<byte>();
            return;
        }

        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            offsets[tetIds[k + 0] + 1]++;
            offsets[tetIds[k + 1] + 1]++;
            offsets[tetIds[k + 2] + 1]++;
            offsets[tetIds[k + 3] + 1]++;
        }

        for (int i = 0; i < numParticles; i++)
            offsets[i + 1] += offsets[i];

        int total = offsets[numParticles];
        tetIndex = new int[total];
        role = new byte[total];

        int[] cur = new int[numParticles];
        Array.Copy(offsets, cur, numParticles);

        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            int a = tetIds[k + 0];
            int b = tetIds[k + 1];
            int c = tetIds[k + 2];
            int d = tetIds[k + 3];

            int ka = cur[a]++; tetIndex[ka] = t; role[ka] = 0;
            int kb = cur[b]++; tetIndex[kb] = t; role[kb] = 1;
            int kc = cur[c]++; tetIndex[kc] = t; role[kc] = 2;
            int kd = cur[d]++; tetIndex[kd] = t; role[kd] = 3;
        }
    }
}
