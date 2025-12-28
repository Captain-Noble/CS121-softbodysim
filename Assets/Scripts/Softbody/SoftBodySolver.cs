// SoftBodySolver.cs
// 请以 UTF-8 保存本文件
using Hanzzz.Tetrahedralizer;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public sealed class SoftBodySolver : MonoBehaviour
{
    #region Types
    public enum ThreadingMode
    {
        InheritManager = 0,
        ForceSingleThread = 1,
        ForceMultithread = 2
    }

    private readonly struct EdgeKey : IEquatable<EdgeKey>
    {
        public readonly int u;
        public readonly int v;
        public EdgeKey(int a, int b)
        {
            if (a < b) { u = a; v = b; }
            else { u = b; v = a; }
        }
        public bool Equals(EdgeKey other) => u == other.u && v == other.v;
        public override bool Equals(object obj) => obj is EdgeKey other && Equals(other);
        public override int GetHashCode() => (u * 73856093) ^ (v * 19349663);
    }

    private readonly struct FaceKey : IEquatable<FaceKey>
    {
        public readonly int a;
        public readonly int b;
        public readonly int c;
        public FaceKey(int i0, int i1, int i2)
        {
            int x = i0, y = i1, z = i2;
            if (x > y) (x, y) = (y, x);
            if (y > z) (y, z) = (z, y);
            if (x > y) (x, y) = (y, x);
            a = x; b = y; c = z;
        }
        public bool Equals(FaceKey other) => a == other.a && b == other.b && c == other.c;
        public override bool Equals(object obj) => obj is FaceKey other && Equals(other);
        public override int GetHashCode() => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
    }

    private struct FaceVal
    {
        public int i0, i1, i2;
        public int opp;
        public int count;
    }
    #endregion

    #region Inspector
    [SerializeField] private bool autoRegisterToManager = true;
    [SerializeField] private ThreadingMode threading = ThreadingMode.InheritManager;

    [SerializeField] private bool autoGenerateAssetFromMesh = true;
    [SerializeField] private MeshFilter sourceMeshFilter;
    [SerializeField] private TetMeshAsset meshAsset;

    [SerializeField] private bool remapVertexData = false;
    [SerializeField][Range(0f, 1f)] private float degenerateTetrahedronRatio = 0f;

    [SerializeField] private float edgeCompliance = 100f;
    [SerializeField] private float volumeCompliance = 0f;

    [Header("Parallel Jacobi (constraint averaging + SOR)")]
    [Tooltip("论文建议 1~2。越大越快但可能更抖/更不稳。")]
    [SerializeField][Range(0.5f, 2.5f)] private float sorOmega = 1.5f;

    [SerializeField] private bool pinTopLayer = false;
    [SerializeField] private float pinTopEpsilon = 1e-4f;

    [SerializeField] private bool groundCollision = true;
    [SerializeField] private float groundY = 0f;

    [SerializeField] private bool colliderCollision = false;
    [SerializeField] private LayerMask colliderMask = ~0;
    [SerializeField] private float particleRadius = 0.02f;

    [SerializeField] private bool updateNormals = true;
    #endregion

    #region State
    private int numParticles;

    private Vector3[] posL;
    private Vector3[] prevPosL;
    private Vector3[] velL;
    private float[] invMass;

    private int[] edgeIds;
    private float[] restEdgeLen;

    private int[] tetIds;
    private float[] restVol;

    private int[] surfaceTris;
    private Vector3[] renderVerts;
    private Mesh mesh;

    // adjacency
    private int[] edgesAdjOffsets;
    private int[] edgesAdjOther;
    private int[] edgesAdjEdge;

    private int[] tetsAdjOffsets;
    private int[] tetsAdjTet;
    private byte[] tetsAdjRole;

    // scratch for Jacobi averaging
    private Vector3[] scratchDelta;
    private int[] scratchCount;

    private readonly Collider[] overlapHits = new Collider[32];

    private Matrix4x4 cachedLocalToWorld;
    private Matrix4x4 cachedWorldToLocal;
    private Vector3 cachedGravityL;

    private bool useMT;
    private ParallelOptions opt = new ParallelOptions();
    #endregion

    #region Unity
    private void OnEnable()
    {
        EnsureAsset();
        BuildFromAsset();
        TryRegister();
    }

    private void Start()
    {
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
        mgr = FindObjectOfType<SoftBodyManager>();
        if (mgr != null) mgr.Register(this);
    }
    #endregion

    #region Build
    private void BuildFromAsset()
    {
        if (meshAsset == null || !meshAsset.IsValid()) throw new InvalidOperationException("Invalid TetMeshAsset");

        numParticles = meshAsset.vertices.Length;

        posL = new Vector3[numParticles];
        prevPosL = new Vector3[numParticles];
        velL = new Vector3[numParticles];
        invMass = new float[numParticles];

        for (int i = 0; i < numParticles; i++)
        {
            Vector3 p = meshAsset.vertices[i];
            posL[i] = p;
            prevPosL[i] = p;
            velL[i] = Vector3.zero;
            invMass[i] = 0f;
        }

        tetIds = (int[])meshAsset.tetIds.Clone();
        int tetCount = tetIds.Length / 4;
        restVol = new float[tetCount];

        for (int i = 0; i < tetCount; i++)
        {
            int a = tetIds[4 * i + 0];
            int b = tetIds[4 * i + 1];
            int c = tetIds[4 * i + 2];
            int d = tetIds[4 * i + 3];

            float vol = SoftBodyParallel.TetVolume(posL[a], posL[b], posL[c], posL[d]);
            restVol[i] = vol;

            float mvol = Mathf.Abs(vol);
            if (mvol > 1e-12f)
            {
                float pInv = 4f / mvol;
                invMass[a] += pInv;
                invMass[b] += pInv;
                invMass[c] += pInv;
                invMass[d] += pInv;
            }
        }

        if (pinTopLayer)
        {
            float topY = float.NegativeInfinity;
            for (int i = 0; i < numParticles; i++) if (posL[i].y > topY) topY = posL[i].y;
            for (int i = 0; i < numParticles; i++)
                if (Mathf.Abs(posL[i].y - topY) <= pinTopEpsilon) invMass[i] = 0f;
        }

        edgeIds = (int[])meshAsset.edgeIds.Clone();
        int edgeCount = edgeIds.Length / 2;
        restEdgeLen = new float[edgeCount];
        for (int i = 0; i < edgeCount; i++)
        {
            int a = edgeIds[2 * i + 0];
            int b = edgeIds[2 * i + 1];
            restEdgeLen[i] = (posL[b] - posL[a]).magnitude;
        }

        surfaceTris = (int[])meshAsset.surfaceTriIds.Clone();

        // Build adjacency (one-time)
        SoftBodyParallel.BuildEdgeAdjacency(numParticles, edgeIds, out edgesAdjOffsets, out edgesAdjOther, out edgesAdjEdge);
        SoftBodyParallel.BuildTetAdjacency(numParticles, tetIds, out tetsAdjOffsets, out tetsAdjTet, out tetsAdjRole);

        // Scratch
        scratchDelta = new Vector3[numParticles];
        scratchCount = new int[numParticles];

        mesh = new Mesh();
        mesh.indexFormat = numParticles > 65535 ? UnityEngine.Rendering.IndexFormat.UInt32 : UnityEngine.Rendering.IndexFormat.UInt16;
        mesh.name = "SoftBodySurface";

        renderVerts = new Vector3[numParticles];
        Array.Copy(posL, renderVerts, numParticles);

        mesh.vertices = renderVerts;
        mesh.triangles = surfaceTris;
        mesh.RecalculateBounds();
        if (updateNormals) mesh.RecalculateNormals();

        GetComponent<MeshFilter>().sharedMesh = mesh;
    }
    #endregion

    #region Step Cache
    public void CacheStepDataMainThread(Vector3 gravityW, bool managerWantsMT, int threads)
    {
        cachedLocalToWorld = transform.localToWorldMatrix;
        cachedWorldToLocal = transform.worldToLocalMatrix;
        cachedGravityL = cachedWorldToLocal.MultiplyVector(gravityW);

        useMT = ResolveUseMT(managerWantsMT);
        opt.MaxDegreeOfParallelism = threads > 0 ? threads : System.Environment.ProcessorCount;
    }

    private bool ResolveUseMT(bool managerWantsMT)
    {
        if (!isActiveAndEnabled) return false;
        if (threading == ThreadingMode.ForceSingleThread) return false;
        if (threading == ThreadingMode.ForceMultithread) return true;
        return managerWantsMT;
    }
    #endregion

    #region Worker Safe
    public void PreSolveWorkerSafe(float dt)
    {
        if (!useMT)
        {
            SoftBodyParallel.PreSolveSerial(posL, prevPosL, velL, invMass, dt, cachedGravityL,
                groundCollision, groundY, cachedLocalToWorld, cachedWorldToLocal);
            return;
        }

        SoftBodyParallel.PreSolveParallel(posL, prevPosL, velL, invMass, dt, cachedGravityL,
            groundCollision, groundY, cachedLocalToWorld, cachedWorldToLocal, opt);
    }

    public void SolveWorkerSafe(float dt)
    {
        if (!useMT)
        {
            // original serial Gauss-Seidel
            SolveEdgesSerial(posL, invMass, edgeIds, restEdgeLen, edgeCompliance, dt);
            SolveVolumesSerial(posL, invMass, tetIds, restVol, volumeCompliance, dt);
            if (groundCollision) SoftBodyParallel.SolveGroundSerial(posL, invMass, groundY, cachedLocalToWorld, cachedWorldToLocal);
            return;
        }

        // NVIDIA-style: group = edges, then group = volumes; each group: gather deltas then apply averaged with omega
        SoftBodyParallel.SolveEdgesJacobiAveraged(
            posL, invMass,
            edgeIds, restEdgeLen,
            edgesAdjOffsets, edgesAdjOther, edgesAdjEdge,
            edgeCompliance, dt, sorOmega,
            scratchDelta, scratchCount,
            opt);

        SoftBodyParallel.SolveVolumesJacobiAveraged(
            posL, invMass,
            tetIds, restVol,
            tetsAdjOffsets, tetsAdjTet, tetsAdjRole,
            volumeCompliance, dt, sorOmega,
            scratchDelta, scratchCount,
            opt);

        if (groundCollision) SoftBodyParallel.SolveGroundSerial(posL, invMass, groundY, cachedLocalToWorld, cachedWorldToLocal);
    }

    public void PostSolveWorkerSafe(float dt)
    {
        if (!useMT)
        {
            SoftBodyParallel.PostSolveSerial(posL, prevPosL, velL, invMass, dt, renderVerts);
            return;
        }

        SoftBodyParallel.PostSolveParallel(posL, prevPosL, velL, invMass, dt, renderVerts, opt);
    }
    #endregion

    #region Main Thread Only
    public void SolveCollidersMainThread()
    {
        if (!colliderCollision) return;
        SolveCollidersMainThreadInternal();
    }

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
    #endregion

    #region Collisions
    private void SolveCollidersMainThreadInternal()
    {
        float r = Mathf.Max(0.0001f, particleRadius);
        Collider selfCol = GetComponent<Collider>();

        for (int i = 0; i < numParticles; i++)
        {
            if (invMass[i] == 0f) continue;

            Vector3 pw = cachedLocalToWorld.MultiplyPoint3x4(posL[i]);

            int hitCount = Physics.OverlapSphereNonAlloc(pw, r, overlapHits, colliderMask, QueryTriggerInteraction.Ignore);
            if (hitCount <= 0) continue;

            for (int h = 0; h < hitCount; h++)
            {
                Collider col = overlapHits[h];
                if (col == null) continue;
                if (selfCol != null && col == selfCol) continue;

                Vector3 cp = col.ClosestPoint(pw);
                Vector3 v = pw - cp;
                float d2 = v.sqrMagnitude;

                if (d2 < r * r)
                {
                    float d = Mathf.Sqrt(Mathf.Max(d2, 1e-12f));
                    Vector3 n = v / d;
                    float push = r - d;
                    pw += n * push;
                }
            }

            posL[i] = cachedWorldToLocal.MultiplyPoint3x4(pw);
        }
    }
    #endregion

    #region Serial Solvers (kept for ST mode)
    private static void SolveEdgesSerial(
        Vector3[] posL,
        float[] invMass,
        int[] edgeIds,
        float[] restEdgeLen,
        float compliance,
        float dt)
    {
        float alpha = dt > 0f ? (compliance / (dt * dt)) : 0f;
        int m = restEdgeLen.Length;

        for (int i = 0; i < m; i++)
        {
            int id0 = edgeIds[2 * i + 0];
            int id1 = edgeIds[2 * i + 1];

            float w0 = invMass[id0];
            float w1 = invMass[id1];
            float w = w0 + w1;
            if (w == 0f) continue;

            Vector3 d = posL[id0] - posL[id1];
            float len = d.magnitude;
            if (len == 0f) continue;

            Vector3 n = d / len;
            float C = len - restEdgeLen[i];
            float s = -C / (w + alpha);

            posL[id0] += n * (s * w0);
            posL[id1] -= n * (s * w1);
        }
    }

    private static void SolveVolumesSerial(
        Vector3[] posL,
        float[] invMass,
        int[] tetIds,
        float[] restVol,
        float compliance,
        float dt)
    {
        float alpha = dt > 0f ? (compliance / (dt * dt)) : 0f;
        int m = restVol.Length;

        for (int i = 0; i < m; i++)
        {
            int a = tetIds[4 * i + 0];
            int b = tetIds[4 * i + 1];
            int c = tetIds[4 * i + 2];
            int d = tetIds[4 * i + 3];

            float wa = invMass[a];
            float wb = invMass[b];
            float wc = invMass[c];
            float wd = invMass[d];

            if (wa + wb + wc + wd == 0f) continue;

            Vector3 pa = posL[a];
            Vector3 pb = posL[b];
            Vector3 pc = posL[c];
            Vector3 pd = posL[d];

            Vector3 ga = Vector3.Cross(pd - pb, pc - pb) / 6f;
            Vector3 gb = Vector3.Cross(pc - pa, pd - pa) / 6f;
            Vector3 gc = Vector3.Cross(pd - pa, pb - pa) / 6f;
            Vector3 gd = Vector3.Cross(pb - pa, pc - pa) / 6f;

            float wsum =
                wa * Vector3.Dot(ga, ga) +
                wb * Vector3.Dot(gb, gb) +
                wc * Vector3.Dot(gc, gc) +
                wd * Vector3.Dot(gd, gd);

            if (wsum == 0f) continue;

            float vol = SoftBodyParallel.TetVolume(pa, pb, pc, pd);
            float C = vol - restVol[i];
            float s = -C / (wsum + alpha);

            posL[a] += ga * (s * wa);
            posL[b] += gb * (s * wb);
            posL[c] += gc * (s * wc);
            posL[d] += gd * (s * wd);
        }
    }
    #endregion

    #region Asset Generator (unchanged)
    private void EnsureAsset()
    {
        if (meshAsset != null && meshAsset.IsValid()) return;
        if (!autoGenerateAssetFromMesh) throw new InvalidOperationException("TetMeshAsset missing");
        if (sourceMeshFilter == null) sourceMeshFilter = GetComponent<MeshFilter>();
        if (sourceMeshFilter == null) throw new InvalidOperationException("MeshFilter missing");
        Mesh src = sourceMeshFilter.sharedMesh;
        if (src == null) throw new InvalidOperationException("Source mesh missing");
        if (!TetAssetGenerator.TryCreateFromMesh(src, remapVertexData, degenerateTetrahedronRatio, out TetMeshAsset generated))
            throw new InvalidOperationException("Tetrahedralization failed");
        meshAsset = generated;
    }

    private static class TetAssetGenerator
    {
        public static bool TryCreateFromMesh(Mesh src, bool remap, float degenerateRatio, out TetMeshAsset asset)
        {
            asset = null;
            if (remap) return false;

            Tetrahedralizer tetrahedralizer = new Tetrahedralizer();
            tetrahedralizer.SetSettings(new Tetrahedralizer.Settings(false, (double)degenerateRatio));

            TetrahedralizedMesh tetrahedralized = ScriptableObject.CreateInstance<TetrahedralizedMesh>();
            TetrahedralMesh tetrahedral = ScriptableObject.CreateInstance<TetrahedralMesh>();

            tetrahedralizer.MeshToTetrahedralizedMesh(src, tetrahedralized);
            tetrahedralizer.TetrahedralizedMeshToTetrahedralMesh(tetrahedralized, tetrahedral);

            if (tetrahedral.vertices == null || tetrahedral.vertices.Count == 0) return false;
            if (tetrahedral.tetrahedrons == null || tetrahedral.tetrahedrons.Count == 0) return false;
            if ((tetrahedral.tetrahedrons.Count & 3) != 0) return false;

            Vector3[] verts = tetrahedral.vertices.ToArray();
            int[] tetIds = tetrahedral.tetrahedrons.ToArray();

            BuildEdgesAndSurface(verts, tetIds, out int[] edgeIds, out int[] surfaceTriIds);

            TetMeshAsset inst = ScriptableObject.CreateInstance<TetMeshAsset>();
            inst.vertices = verts;
            inst.tetIds = tetIds;
            inst.edgeIds = edgeIds;
            inst.surfaceTriIds = surfaceTriIds;

#if UNITY_EDITOR
            string folderRoot = "Assets/SoftBody";
            string folderGen = "Assets/SoftBody/Generated";

            if (!UnityEditor.AssetDatabase.IsValidFolder(folderRoot))
            {
                if (!UnityEditor.AssetDatabase.IsValidFolder("Assets")) return false;
                UnityEditor.AssetDatabase.CreateFolder("Assets", "SoftBody");
            }

            if (!UnityEditor.AssetDatabase.IsValidFolder(folderGen))
            {
                UnityEditor.AssetDatabase.CreateFolder(folderRoot, "Generated");
            }

            string safeName = string.IsNullOrEmpty(src.name) ? "Mesh" : src.name;
            string path = UnityEditor.AssetDatabase.GenerateUniqueAssetPath($"{folderGen}/{safeName}_Tet.asset");
            UnityEditor.AssetDatabase.CreateAsset(inst, path);
            UnityEditor.AssetDatabase.SaveAssets();
            UnityEditor.AssetDatabase.Refresh();
            asset = UnityEditor.AssetDatabase.LoadAssetAtPath<TetMeshAsset>(path);
            return asset != null;
#else
            inst.hideFlags = HideFlags.DontSaveInBuild | HideFlags.DontSaveInEditor;
            asset = inst;
            return true;
#endif
        }

        private static void BuildEdgesAndSurface(Vector3[] verts, int[] tetIds, out int[] edgeIds, out int[] surfaceTriIds)
        {
            int tetCount = tetIds.Length / 4;

            var edgeSet = new HashSet<EdgeKey>(tetCount * 8);
            var faces = new Dictionary<FaceKey, FaceVal>(tetCount * 4);

            void AddEdge(int a, int b) => edgeSet.Add(new EdgeKey(a, b));

            void AddFace(int i0, int i1, int i2, int opp)
            {
                var key = new FaceKey(i0, i1, i2);
                if (faces.TryGetValue(key, out var v))
                {
                    v.count++;
                    faces[key] = v;
                }
                else
                {
                    faces[key] = new FaceVal { i0 = i0, i1 = i1, i2 = i2, opp = opp, count = 1 };
                }
            }

            for (int t = 0; t < tetCount; t++)
            {
                int k = t * 4;
                int a = tetIds[k + 0];
                int b = tetIds[k + 1];
                int c = tetIds[k + 2];
                int d = tetIds[k + 3];

                AddEdge(a, b); AddEdge(a, c); AddEdge(a, d);
                AddEdge(b, c); AddEdge(b, d);
                AddEdge(c, d);

                AddFace(a, b, c, d);
                AddFace(a, d, b, c);
                AddFace(a, c, d, b);
                AddFace(b, d, c, a);
            }

            edgeIds = new int[edgeSet.Count * 2];
            int ei = 0;
            foreach (var e in edgeSet)
            {
                edgeIds[ei++] = e.u;
                edgeIds[ei++] = e.v;
            }

            var tris = new List<int>(faces.Count * 3);

            foreach (var kv in faces)
            {
                FaceVal f = kv.Value;
                if (f.count != 1) continue;

                int i0 = f.i0;
                int i1 = f.i1;
                int i2 = f.i2;
                int opp = f.opp;

                Vector3 p0 = verts[i0];
                Vector3 p1 = verts[i1];
                Vector3 p2 = verts[i2];
                Vector3 po = verts[opp];

                Vector3 n = Vector3.Cross(p1 - p0, p2 - p0);
                float s = Vector3.Dot(n, po - p0);

                if (s > 0f)
                {
                    int tmp = i1;
                    i1 = i2;
                    i2 = tmp;
                }

                tris.Add(i0);
                tris.Add(i1);
                tris.Add(i2);
            }

            surfaceTriIds = tris.ToArray();
        }
    }
    #endregion
}
