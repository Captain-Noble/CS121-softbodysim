// SoftBodySolver.cs
using System;
using System.Runtime.InteropServices;
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

    [Header("GPU Backend")]
    [SerializeField] private bool allowGpu = true;
    [SerializeField] private ComputeShader softBodyCompute;
    [SerializeField] private bool enableGpuRendering = true;
    [SerializeField] private Material gpuRenderMaterialOverride;

    [Header("Mesh / Asset")]
    [SerializeField] private bool autoGenerateAssetFromMesh = true;
    [SerializeField] private MeshFilter sourceMeshFilter;
    [SerializeField] private SoftBodyTetMeshAsset meshAsset;
    [SerializeField, Range(0f, 1f)] private float degenerateTetrahedronRatio = 0f;

    [Header("Constraints")]
    [SerializeField, Range(0f, 1f)] private float edgeStiffness = 0.9f;
    [SerializeField, Range(0f, 1f)] private float volumeStiffness = 0.98f;

    [Header("Jacobi + SOR")]
    [SerializeField, Range(0.5f, 2.0f)] private float sorOmega = 1.4f;

    [Header("Mass")]
    [SerializeField] private float density = 1f;

    [Header("Pinning")]
    [SerializeField] private bool pinTopLayer = false;
    [SerializeField] private float pinTopEpsilon = 1e-4f;

    [Header("Collisions")]
    [SerializeField] private bool primitiveCollision = true;
    [SerializeField] private float particleRadius = 0.02f;

    [Header("Ground")]
    [SerializeField] private bool collideWithGround = true;
    [SerializeField] private Transform groundTransform;
    [SerializeField] private float groundHeight = 0f;
    [SerializeField] private Vector3 groundNormal = Vector3.up;

    [Header("Rendering")]
    [SerializeField] private bool updateNormals = true;

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

    private int[] edgesAdjOffsets, edgesAdjOther, edgesAdjEdge;
    private int[] tetsAdjOffsets, tetsAdjTet, tetsAdjRoleI;

    private int[] triAdjOffsets;
    private int[] triAdjTri;

    private Vector3[] scratchDelta;
    private int[] scratchCount;

    private Mesh mesh;
    private Vector3[] renderVerts;

    private Matrix4x4 cachedLocalToWorld;
    private Matrix4x4 cachedWorldToLocal;
    private Vector3 cachedGravityL;

    private SoftBodyPrimitiveCollider.PrimitiveColliderData[] cachedColliders;
    private int cachedColliderCount;

    private bool cachedGroundEnabled;
    private Vector3 cachedGroundPointW;
    private Vector3 cachedGroundNormalW;

    private bool useParallel;
    private bool useGpu;
    private int cachedSolverIterations;

    private readonly ParallelOptions opt = new ParallelOptions();

    [StructLayout(LayoutKind.Sequential)]
    private struct GpuCollider
    {
        public int type;
        public Vector3 positionW;
        public float pad0;
        public Vector4 rotationW;
        public Vector3 data;
        public float pad1;
    }

    private sealed class GpuBackend
    {
        public ComputeShader cs;
        public int kPre, kEdgeGather, kVolGather, kApply, kCollide, kPost, kNormals;

        public ComputeBuffer pos;
        public ComputeBuffer posPrev;
        public ComputeBuffer posPred;
        public ComputeBuffer vel;
        public ComputeBuffer invMass;

        public ComputeBuffer edgeAdjOffsets, edgeAdjOther, edgeAdjEdge, restEdgeLen;
        public ComputeBuffer tetAdjOffsets, tetAdjTet, tetAdjRole, tetIds, restTetVol;

        public ComputeBuffer scratchDelta;
        public ComputeBuffer scratchCount;

        public ComputeBuffer colliders;
        public int collidersCapacity;

        public ComputeBuffer surfaceTris;
        public ComputeBuffer triAdjOffsets;
        public ComputeBuffer triAdjTri;
        public ComputeBuffer normals;

        public void Release()
        {
            pos?.Release(); pos = null;
            posPrev?.Release(); posPrev = null;
            posPred?.Release(); posPred = null;
            vel?.Release(); vel = null;
            invMass?.Release(); invMass = null;

            edgeAdjOffsets?.Release(); edgeAdjOffsets = null;
            edgeAdjOther?.Release(); edgeAdjOther = null;
            edgeAdjEdge?.Release(); edgeAdjEdge = null;
            restEdgeLen?.Release(); restEdgeLen = null;

            tetAdjOffsets?.Release(); tetAdjOffsets = null;
            tetAdjTet?.Release(); tetAdjTet = null;
            tetAdjRole?.Release(); tetAdjRole = null;
            tetIds?.Release(); tetIds = null;
            restTetVol?.Release(); restTetVol = null;

            scratchDelta?.Release(); scratchDelta = null;
            scratchCount?.Release(); scratchCount = null;

            colliders?.Release(); colliders = null;
            collidersCapacity = 0;

            surfaceTris?.Release(); surfaceTris = null;
            triAdjOffsets?.Release(); triAdjOffsets = null;
            triAdjTri?.Release(); triAdjTri = null;
            normals?.Release(); normals = null;
        }
    }

    private GpuBackend gpu;
    private MaterialPropertyBlock mpb;
    private MeshRenderer cachedRenderer;

    private static int DivUp(int a, int b) => (a + b - 1) / b;

    private void OnEnable()
    {
        EnsureAsset();
        BuildFromAsset();
        TryRegister();
    }

    private void OnDisable()
    {
        if (autoRegisterToManager && SoftBodyManager.Instance != null)
            SoftBodyManager.Instance.Unregister(this);

        gpu?.Release();
        gpu = null;
    }

    private void TryRegister()
    {
        if (!autoRegisterToManager) return;

        var mgr = SoftBodyManager.Instance;
        if (mgr != null) { mgr.Register(this); return; }

        mgr = FindObjectOfType<SoftBodyManager>(true);
        if (mgr != null) mgr.Register(this);
    }

    public void CacheStepDataMainThread(
        Vector3 gravityW,
        SoftBodyManager.ComputeMode mode,
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

        bool managerWantsParallel = (mode == SoftBodyManager.ComputeMode.MultiThreadWithinSolver);
        useParallel = ResolveUseParallel(managerWantsParallel);
        opt.MaxDegreeOfParallelism = threads > 0 ? threads : Environment.ProcessorCount;

        bool managerWantsGpu = (mode == SoftBodyManager.ComputeMode.GpuCompute);
        bool wantGpu = allowGpu && managerWantsGpu && softBodyCompute != null;

        if (wantGpu != useGpu)
        {
            if (useGpu && !wantGpu) SyncGpuToCpuBlocking();
            if (!useGpu && wantGpu)
            {
                EnsureGpuCreated();
                UploadCpuToGpu();
            }
        }
        useGpu = wantGpu;

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

    public void PreSolveWorkerSafe(float dt)
    {
        if (useGpu) { GpuPreSolve(dt); return; }

        if (!useParallel)
        {
            for (int i = 0; i < n; i++) PreSolveOne(i, dt);
        }
        else
        {
            Parallel.For(0, n, opt, i => PreSolveOne(i, dt));
        }
    }

    public void SolveWorkerSafe(float dt)
    {
        if (useGpu) { GpuSolve(dt); return; }

        int iters = cachedSolverIterations;
        for (int iter = 0; iter < iters; iter++)
        {
            SolveEdgesGatherApply();
            SolveVolumesGatherApply();

            if (cachedGroundEnabled || (primitiveCollision && cachedColliderCount > 0))
                SolveAllCollisions();
        }
    }

    public void PostSolveWorkerSafe(float dt)
    {
        if (useGpu) { GpuPostSolve(dt); return; }

        if (!useParallel)
        {
            for (int i = 0; i < n; i++) PostSolveOne(i, dt);
        }
        else
        {
            Parallel.For(0, n, opt, i => PostSolveOne(i, dt));
        }

        Array.Copy(posL, renderVerts, n);
    }

    public void UploadMeshVerticesBoundsMainThread()
    {
        if (mesh == null) return;

        if (useGpu && enableGpuRendering)
        {
            EnsureGpuRenderBinding();
            return;
        }

        mesh.vertices = renderVerts;
        mesh.RecalculateBounds();
    }

    public void UploadMeshNormalsMainThread()
    {
        if (!updateNormals) return;
        if (mesh == null) return;

        if (useGpu && enableGpuRendering)
        {
            EnsureGpuCreated();
            SetGpuCommonConstants(0f);
            BindGpuBuffersForKernels();

            int groups = DivUp(n, 256);
            gpu.cs.Dispatch(gpu.kNormals, groups, 1, 1);

            EnsureGpuRenderBinding();
            return;
        }

        mesh.RecalculateNormals();
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

    private void PostSolveOne(int i, float dt)
    {
        if (invMass[i] == 0f)
        {
            posL[i] = posPrevL[i];
            posPredL[i] = posPrevL[i];
            velL[i] = Vector3.zero;
            return;
        }

        float invDt = dt > 0f ? 1f / dt : 0f;
        velL[i] = (posPredL[i] - posPrevL[i]) * invDt;
        posL[i] = posPredL[i];
    }

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
            int role = tetsAdjRoleI[kk];

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

        posPredL[i] += (sorOmega / cnt) * scratchDelta[i];
    }

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

        if (cachedGroundEnabled)
        {
            float dist = Vector3.Dot(cachedGroundNormalW, pw - cachedGroundPointW);
            if (dist < r) pw += cachedGroundNormalW * (r - dist);
        }

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

    private void EnsureGpuCreated()
    {
        if (gpu != null) return;
        if (softBodyCompute == null) throw new InvalidOperationException("ComputeShader missing");

        gpu = new GpuBackend();
        gpu.cs = softBodyCompute;

        gpu.kPre = gpu.cs.FindKernel("K_PreSolve");
        gpu.kEdgeGather = gpu.cs.FindKernel("K_EdgeGather");
        gpu.kVolGather = gpu.cs.FindKernel("K_VolumeGather");
        gpu.kApply = gpu.cs.FindKernel("K_ApplyDelta");
        gpu.kCollide = gpu.cs.FindKernel("K_Collide");
        gpu.kPost = gpu.cs.FindKernel("K_PostSolve");
        gpu.kNormals = gpu.cs.FindKernel("K_UpdateNormals");

        gpu.pos = new ComputeBuffer(n, sizeof(float) * 3);
        gpu.posPrev = new ComputeBuffer(n, sizeof(float) * 3);
        gpu.posPred = new ComputeBuffer(n, sizeof(float) * 3);
        gpu.vel = new ComputeBuffer(n, sizeof(float) * 3);
        gpu.invMass = new ComputeBuffer(n, sizeof(float));

        gpu.edgeAdjOffsets = new ComputeBuffer(edgesAdjOffsets.Length, sizeof(int));
        gpu.edgeAdjOther = new ComputeBuffer(edgesAdjOther.Length, sizeof(int));
        gpu.edgeAdjEdge = new ComputeBuffer(edgesAdjEdge.Length, sizeof(int));
        gpu.restEdgeLen = new ComputeBuffer(restEdgeLen.Length, sizeof(float));
        gpu.edgeAdjOffsets.SetData(edgesAdjOffsets);
        gpu.edgeAdjOther.SetData(edgesAdjOther);
        gpu.edgeAdjEdge.SetData(edgesAdjEdge);
        gpu.restEdgeLen.SetData(restEdgeLen);

        gpu.tetAdjOffsets = new ComputeBuffer(tetsAdjOffsets.Length, sizeof(int));
        gpu.tetAdjTet = new ComputeBuffer(tetsAdjTet.Length, sizeof(int));
        gpu.tetAdjRole = new ComputeBuffer(tetsAdjRoleI.Length, sizeof(int));
        gpu.tetIds = new ComputeBuffer(tetIds.Length, sizeof(int));
        gpu.restTetVol = new ComputeBuffer(restTetVol.Length, sizeof(float));
        gpu.tetAdjOffsets.SetData(tetsAdjOffsets);
        gpu.tetAdjTet.SetData(tetsAdjTet);
        gpu.tetAdjRole.SetData(tetsAdjRoleI);
        gpu.tetIds.SetData(tetIds);
        gpu.restTetVol.SetData(restTetVol);

        gpu.scratchDelta = new ComputeBuffer(n, sizeof(float) * 3);
        gpu.scratchCount = new ComputeBuffer(n, sizeof(int));

        gpu.collidersCapacity = 64;
        gpu.colliders = new ComputeBuffer(gpu.collidersCapacity, Marshal.SizeOf(typeof(GpuCollider)));

        gpu.surfaceTris = new ComputeBuffer(surfaceTris.Length, sizeof(int));
        gpu.surfaceTris.SetData(surfaceTris);

        gpu.triAdjOffsets = new ComputeBuffer(triAdjOffsets.Length, sizeof(int));
        gpu.triAdjOffsets.SetData(triAdjOffsets);

        gpu.triAdjTri = new ComputeBuffer(triAdjTri.Length, sizeof(int));
        gpu.triAdjTri.SetData(triAdjTri);

        gpu.normals = new ComputeBuffer(n, sizeof(float) * 3);

        EnsureGpuRenderBinding();
    }

    private void UploadCpuToGpu()
    {
        EnsureGpuCreated();
        gpu.pos.SetData(posL);
        gpu.posPrev.SetData(posPrevL);
        gpu.posPred.SetData(posPredL);
        gpu.vel.SetData(velL);
        gpu.invMass.SetData(invMass);
    }

    private void SyncGpuToCpuBlocking()
    {
        if (gpu == null) return;
        gpu.pos.GetData(posL);
        gpu.posPrev.GetData(posPrevL);
        gpu.posPred.GetData(posPredL);
        gpu.vel.GetData(velL);
        Array.Copy(posL, renderVerts, n);
    }

    private void EnsureGpuCollidersUploaded()
    {
        if (gpu == null) return;

        int count = (primitiveCollision ? cachedColliderCount : 0);
        if (count <= 0)
        {
            gpu.cs.SetInt("_ColliderCount", 0);
            return;
        }

        if (count > gpu.collidersCapacity)
        {
            gpu.colliders.Release();
            gpu.collidersCapacity = Mathf.NextPowerOfTwo(count);
            gpu.colliders = new ComputeBuffer(gpu.collidersCapacity, Marshal.SizeOf(typeof(GpuCollider)));
            EnsureGpuRenderBinding();
        }

        var tmp = new GpuCollider[count];
        for (int i = 0; i < count; i++)
        {
            var c = cachedColliders[i];
            tmp[i].type = (int)c.type;
            tmp[i].positionW = c.positionW;
            tmp[i].pad0 = 0f;
            tmp[i].rotationW = new Vector4(c.rotationW.x, c.rotationW.y, c.rotationW.z, c.rotationW.w);
            tmp[i].data = c.data;
            tmp[i].pad1 = 0f;
        }

        gpu.colliders.SetData(tmp, 0, 0, count);
        gpu.cs.SetInt("_ColliderCount", count);
    }

    private void SetGpuCommonConstants(float dt)
    {
        var cs = gpu.cs;

        cs.SetInt("_N", n);
        cs.SetFloat("_Dt", dt);

        cs.SetFloat("_EdgeStiffness", edgeStiffness);
        cs.SetFloat("_VolumeStiffness", volumeStiffness);
        cs.SetFloat("_SorOmega", sorOmega);

        cs.SetVector("_GravityL", cachedGravityL);

        cs.SetFloat("_ParticleRadius", Mathf.Max(1e-6f, particleRadius));
        cs.SetInt("_PrimitiveCollisionEnabled", (primitiveCollision ? 1 : 0));

        cs.SetInt("_GroundEnabled", cachedGroundEnabled ? 1 : 0);
        cs.SetVector("_GroundPointW", cachedGroundPointW);
        cs.SetVector("_GroundNormalW", cachedGroundNormalW);

        cs.SetMatrix("_LocalToWorld", cachedLocalToWorld);
        cs.SetMatrix("_WorldToLocal", cachedWorldToLocal);
    }

    private void BindGpuBuffersForKernels()
    {
        var cs = gpu.cs;

        cs.SetBuffer(gpu.kPre, "_Pos", gpu.pos);
        cs.SetBuffer(gpu.kPre, "_PosPrev", gpu.posPrev);
        cs.SetBuffer(gpu.kPre, "_PosPred", gpu.posPred);
        cs.SetBuffer(gpu.kPre, "_Vel", gpu.vel);
        cs.SetBuffer(gpu.kPre, "_InvMass", gpu.invMass);

        cs.SetBuffer(gpu.kEdgeGather, "_PosPred", gpu.posPred);
        cs.SetBuffer(gpu.kEdgeGather, "_InvMass", gpu.invMass);
        cs.SetBuffer(gpu.kEdgeGather, "_EdgeAdjOffsets", gpu.edgeAdjOffsets);
        cs.SetBuffer(gpu.kEdgeGather, "_EdgeAdjOther", gpu.edgeAdjOther);
        cs.SetBuffer(gpu.kEdgeGather, "_EdgeAdjEdge", gpu.edgeAdjEdge);
        cs.SetBuffer(gpu.kEdgeGather, "_RestEdgeLen", gpu.restEdgeLen);
        cs.SetBuffer(gpu.kEdgeGather, "_ScratchDelta", gpu.scratchDelta);
        cs.SetBuffer(gpu.kEdgeGather, "_ScratchCount", gpu.scratchCount);

        cs.SetBuffer(gpu.kVolGather, "_PosPred", gpu.posPred);
        cs.SetBuffer(gpu.kVolGather, "_InvMass", gpu.invMass);
        cs.SetBuffer(gpu.kVolGather, "_TetAdjOffsets", gpu.tetAdjOffsets);
        cs.SetBuffer(gpu.kVolGather, "_TetAdjTet", gpu.tetAdjTet);
        cs.SetBuffer(gpu.kVolGather, "_TetAdjRole", gpu.tetAdjRole);
        cs.SetBuffer(gpu.kVolGather, "_TetIds", gpu.tetIds);
        cs.SetBuffer(gpu.kVolGather, "_RestTetVol", gpu.restTetVol);
        cs.SetBuffer(gpu.kVolGather, "_ScratchDelta", gpu.scratchDelta);
        cs.SetBuffer(gpu.kVolGather, "_ScratchCount", gpu.scratchCount);

        cs.SetBuffer(gpu.kApply, "_PosPred", gpu.posPred);
        cs.SetBuffer(gpu.kApply, "_InvMass", gpu.invMass);
        cs.SetBuffer(gpu.kApply, "_ScratchDelta", gpu.scratchDelta);
        cs.SetBuffer(gpu.kApply, "_ScratchCount", gpu.scratchCount);

        cs.SetBuffer(gpu.kCollide, "_PosPred", gpu.posPred);
        cs.SetBuffer(gpu.kCollide, "_InvMass", gpu.invMass);
        cs.SetBuffer(gpu.kCollide, "_Colliders", gpu.colliders);

        cs.SetBuffer(gpu.kPost, "_Pos", gpu.pos);
        cs.SetBuffer(gpu.kPost, "_PosPrev", gpu.posPrev);
        cs.SetBuffer(gpu.kPost, "_PosPred", gpu.posPred);
        cs.SetBuffer(gpu.kPost, "_Vel", gpu.vel);
        cs.SetBuffer(gpu.kPost, "_InvMass", gpu.invMass);

        cs.SetBuffer(gpu.kNormals, "_Pos", gpu.pos);
        cs.SetBuffer(gpu.kNormals, "_SurfaceTris", gpu.surfaceTris);
        cs.SetBuffer(gpu.kNormals, "_TriAdjOffsets", gpu.triAdjOffsets);
        cs.SetBuffer(gpu.kNormals, "_TriAdjTri", gpu.triAdjTri);
        cs.SetBuffer(gpu.kNormals, "_Normals", gpu.normals);
    }

    private void GpuPreSolve(float dt)
    {
        EnsureGpuCreated();
        SetGpuCommonConstants(dt);
        BindGpuBuffersForKernels();

        int groups = DivUp(n, 256);
        gpu.cs.Dispatch(gpu.kPre, groups, 1, 1);
    }

    private void GpuSolve(float dt)
    {
        EnsureGpuCreated();
        SetGpuCommonConstants(dt);
        BindGpuBuffersForKernels();
        EnsureGpuCollidersUploaded();

        int groups = DivUp(n, 256);
        int iters = cachedSolverIterations;

        for (int iter = 0; iter < iters; iter++)
        {
            gpu.cs.Dispatch(gpu.kEdgeGather, groups, 1, 1);
            gpu.cs.Dispatch(gpu.kApply, groups, 1, 1);

            gpu.cs.Dispatch(gpu.kVolGather, groups, 1, 1);
            gpu.cs.Dispatch(gpu.kApply, groups, 1, 1);

            if (cachedGroundEnabled || (primitiveCollision && cachedColliderCount > 0))
                gpu.cs.Dispatch(gpu.kCollide, groups, 1, 1);
        }
    }

    private void GpuPostSolve(float dt)
    {
        EnsureGpuCreated();
        SetGpuCommonConstants(dt);
        BindGpuBuffersForKernels();

        int groups = DivUp(n, 256);
        gpu.cs.Dispatch(gpu.kPost, groups, 1, 1);

        if (!enableGpuRendering)
            SyncGpuToCpuBlocking();
    }

    private void EnsureGpuRenderBinding()
    {
        if (!enableGpuRendering) return;
        if (gpu == null || gpu.pos == null || gpu.normals == null) return;

        if (cachedRenderer == null) cachedRenderer = GetComponent<MeshRenderer>();
        if (cachedRenderer == null) return;

        if (gpuRenderMaterialOverride != null)
            cachedRenderer.sharedMaterial = gpuRenderMaterialOverride;

        if (mpb == null) mpb = new MaterialPropertyBlock();
        cachedRenderer.GetPropertyBlock(mpb);

        mpb.SetBuffer("_Positions", gpu.pos);
        mpb.SetBuffer("_Normals", gpu.normals);

        var sun = RenderSettings.sun;
        Vector3 lightDir = Vector3.down;
        Vector3 lightColor = Vector3.one;

        if (sun != null && sun.type == LightType.Directional)
        {
            lightDir = -sun.transform.forward;
            Color c = sun.color * sun.intensity;
            lightColor = new Vector3(c.r, c.g, c.b);
        }

        mpb.SetVector("_LightDirWS", lightDir.normalized);
        mpb.SetVector("_LightColor", lightColor);

        cachedRenderer.SetPropertyBlock(mpb);

        if (mesh != null)
            mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 10000f);
    }

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

            restTetVol[t] = signedVol;
        }

        for (int i = 0; i < n; i++)
            invMass[i] = mass[i] > 0f ? 1f / mass[i] : 0f;

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
        BuildTetAdjacency(n, tetIds, out tetsAdjOffsets, out tetsAdjTet, out tetsAdjRoleI);
        BuildTriAdjacency(n, surfaceTris, out triAdjOffsets, out triAdjTri);

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

    private static float TetSignedVolume(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
    {
        Vector3 a = p1 - p0;
        Vector3 b = p2 - p0;
        Vector3 c = p3 - p0;
        return Vector3.Dot(Vector3.Cross(a, b), c) / 6f;
    }

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
        out int[] roleI)
    {
        int tetCount = tetIds != null ? (tetIds.Length / 4) : 0;
        offsets = new int[numParticles + 1];

        if (tetCount == 0)
        {
            tetIndex = Array.Empty<int>();
            roleI = Array.Empty<int>();
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
        roleI = new int[total];

        int[] cur = new int[numParticles];
        Array.Copy(offsets, cur, numParticles);

        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            int a = tetIds[k + 0];
            int b = tetIds[k + 1];
            int c = tetIds[k + 2];
            int d = tetIds[k + 3];

            int ka = cur[a]++; tetIndex[ka] = t; roleI[ka] = 0;
            int kb = cur[b]++; tetIndex[kb] = t; roleI[kb] = 1;
            int kc = cur[c]++; tetIndex[kc] = t; roleI[kc] = 2;
            int kd = cur[d]++; tetIndex[kd] = t; roleI[kd] = 3;
        }
    }

    private static void BuildTriAdjacency(
        int numParticles,
        int[] surfaceTris,
        out int[] offsets,
        out int[] triIndex)
    {
        int triCount = surfaceTris != null ? (surfaceTris.Length / 3) : 0;
        offsets = new int[numParticles + 1];

        if (triCount == 0)
        {
            triIndex = Array.Empty<int>();
            return;
        }

        for (int t = 0; t < triCount; t++)
        {
            int k = 3 * t;
            offsets[surfaceTris[k + 0] + 1]++;
            offsets[surfaceTris[k + 1] + 1]++;
            offsets[surfaceTris[k + 2] + 1]++;
        }

        for (int i = 0; i < numParticles; i++)
            offsets[i + 1] += offsets[i];

        int total = offsets[numParticles];
        triIndex = new int[total];

        int[] cur = new int[numParticles];
        Array.Copy(offsets, cur, numParticles);

        for (int t = 0; t < triCount; t++)
        {
            int k = 3 * t;
            int a = surfaceTris[k + 0];
            int b = surfaceTris[k + 1];
            int c = surfaceTris[k + 2];

            triIndex[cur[a]++] = t;
            triIndex[cur[b]++] = t;
            triIndex[cur[c]++] = t;
        }
    }
}
