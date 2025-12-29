// PBDSoftBody.cs
using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public sealed class PBDSoftBody : MonoBehaviour
{
    [Header("Auto Tetrahedralize")]
    public bool autoTetrahedralize = true;

    [Tooltip("Remove too-small tets. 0 keeps all. See Tetrahedralizer docs.")]
    [Range(0.0f, 1.0f)]
    public double degenerateTetrahedronRatio = 0.0;

    [Header("Mass / Pinning")]
    public bool pinTopLayer = false;
    public float pinTopEpsilon = 1e-4f;

    [Header("XPBD Compliance (0 = rigid)")]
    public float edgeCompliance = 0.0005f;
    public float volumeCompliance = 0.0f;

    [Header("Rendering")]
    public bool updateNormals = false;
    public int normalsEveryNFrames = 0;

    internal RuntimeTetMesh runtimeMesh;

    internal int globalParticleStart;
    internal int globalParticleCount;
    internal int globalEdgeStart;
    internal int globalEdgeCount;
    internal int globalTetStart;
    internal int globalTetCount;

    private Mesh _mesh;
    private Vector3[] _meshVerts;
    private int[] _surfaceTris;
    private int _frameCounter;

    private void OnEnable()
    {
        if (autoTetrahedralize)
        {
            if (!TryBuildRuntimeTetMesh(out string err))
            {
                Debug.LogError($"[{name}] Auto tetrahedralize failed: {err}", this);
                enabled = false;
                return;
            }
        }

        if (runtimeMesh == null || !runtimeMesh.IsValid)
        {
            Debug.LogError($"[{name}] runtimeMesh is missing/invalid. Enable autoTetrahedralize or set runtimeMesh in code.", this);
            enabled = false;
            return;
        }

        BuildRenderMeshFromTetSurface();

        var world = FindAnyObjectByType<PBDWorld>();
        if (world == null)
        {
            Debug.LogError("No PBDWorld found in scene.");
            enabled = false;
            return;
        }
        world.RegisterBody(this);
    }

    private void OnDisable()
    {
        var world = FindAnyObjectByType<PBDWorld>();
        if (world != null) world.UnregisterBody(this);
    }

    private bool TryBuildRuntimeTetMesh(out string error)
    {
        error = null;

        var mf = GetComponent<MeshFilter>();
        var src = mf != null ? mf.sharedMesh : null;
        if (src == null)
        {
            error = "MeshFilter.sharedMesh is null.";
            return false;
        }

        if (!TetrahedralizerBridge.IsAvailable())
        {
            error = "Tetrahedralizer plugin not detected. (Type 'Tetrahedralizer' not found).";
            return false;
        }

        var tet = TetrahedralizerBridge.GenerateTetMeshFromUnityMesh(src, degenerateTetrahedronRatio, out error);
        if (tet == null) return false;

        BuildEdgesAndSurface(tet.vertices, tet.tetIds, out tet.edgeIds, out tet.surfaceTriIds);

        runtimeMesh = tet;
        return true;
    }

    private void BuildRenderMeshFromTetSurface()
    {
        _mesh = new Mesh();
        _mesh.name = $"{name}_PBDTetSurface";
        _mesh.indexFormat = runtimeMesh.vertices.Length > 65535
            ? UnityEngine.Rendering.IndexFormat.UInt32
            : UnityEngine.Rendering.IndexFormat.UInt16;

        _meshVerts = new Vector3[runtimeMesh.vertices.Length];
        Array.Copy(runtimeMesh.vertices, _meshVerts, _meshVerts.Length);

        _surfaceTris = runtimeMesh.surfaceTriIds;

        _mesh.vertices = _meshVerts;
        _mesh.triangles = _surfaceTris;

        _mesh.RecalculateBounds();
        if (updateNormals) _mesh.RecalculateNormals();
        _mesh.MarkDynamic();

        GetComponent<MeshFilter>().sharedMesh = _mesh;
    }

    internal void UploadFromWorld(Vector3[] worldPositions)
    {
        if (_mesh == null || _meshVerts == null) return;

        var w2l = transform.worldToLocalMatrix;
        int start = globalParticleStart;
        int count = globalParticleCount;

        for (int i = 0; i < count; i++)
        {
            _meshVerts[i] = w2l.MultiplyPoint3x4(worldPositions[start + i]);
        }

        _mesh.vertices = _meshVerts;
        _mesh.RecalculateBounds();

        if (updateNormals)
        {
            if (normalsEveryNFrames <= 1) _mesh.RecalculateNormals();
            else
            {
                _frameCounter++;
                if ((_frameCounter % normalsEveryNFrames) == 0) _mesh.RecalculateNormals();
            }
        }
    }

    internal void GetInitialWorldPositions(Vector3[] dst, int dstStart)
    {
        var l2w = transform.localToWorldMatrix;
        var verts = runtimeMesh.vertices;
        for (int i = 0; i < verts.Length; i++)
        {
            dst[dstStart + i] = l2w.MultiplyPoint3x4(verts[i]);
        }
    }

    internal void ApplyPinning(float[] invMassGlobal, int start)
    {
        if (!pinTopLayer) return;

        float topY = float.NegativeInfinity;
        var verts = runtimeMesh.vertices;
        for (int i = 0; i < verts.Length; i++)
            if (verts[i].y > topY) topY = verts[i].y;

        for (int i = 0; i < verts.Length; i++)
        {
            if (Mathf.Abs(verts[i].y - topY) <= pinTopEpsilon)
                invMassGlobal[start + i] = 0f;
        }
    }

    private struct EdgeKey : IEquatable<EdgeKey>
    {
        public readonly int a;
        public readonly int b;
        public EdgeKey(int i0, int i1)
        {
            if (i0 < i1) { a = i0; b = i1; }
            else { a = i1; b = i0; }
        }
        public bool Equals(EdgeKey other) => a == other.a && b == other.b;
        public override int GetHashCode() => (a * 73856093) ^ (b * 19349663);
    }

    private struct FaceKey : IEquatable<FaceKey>
    {
        public readonly int a, b, c;
        public FaceKey(int i0, int i1, int i2)
        {
            int x = i0, y = i1, z = i2;
            if (x > y) (x, y) = (y, x);
            if (y > z) (y, z) = (z, y);
            if (x > y) (x, y) = (y, x);
            a = x; b = y; c = z;
        }
        public bool Equals(FaceKey other) => a == other.a && b == other.b && c == other.c;
        public override int GetHashCode() => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
    }

    private struct FaceOri
    {
        public int i0, i1, i2;
        public FaceOri(int a, int b, int c) { i0 = a; i1 = b; i2 = c; }
    }

    private static void BuildEdgesAndSurface(
        Vector3[] vertices,
        int[] tetIds,
        out int[] edgeIds,
        out int[] surfaceTriIds)
    {
        var edges = new HashSet<EdgeKey>(tetIds.Length);
        var faceCount = new Dictionary<FaceKey, int>(tetIds.Length);
        var faceOri = new Dictionary<FaceKey, FaceOri>(tetIds.Length);

        int tetCount = tetIds.Length / 4;
        for (int t = 0; t < tetCount; t++)
        {
            int a = tetIds[t * 4 + 0];
            int b = tetIds[t * 4 + 1];
            int c = tetIds[t * 4 + 2];
            int d = tetIds[t * 4 + 3];

            edges.Add(new EdgeKey(a, b));
            edges.Add(new EdgeKey(a, c));
            edges.Add(new EdgeKey(a, d));
            edges.Add(new EdgeKey(b, c));
            edges.Add(new EdgeKey(b, d));
            edges.Add(new EdgeKey(c, d));

            AddFace(faceCount, faceOri, a, b, c);
            AddFace(faceCount, faceOri, a, d, b);
            AddFace(faceCount, faceOri, a, c, d);
            AddFace(faceCount, faceOri, b, d, c);
        }

        edgeIds = new int[edges.Count * 2];
        int ei = 0;
        foreach (var e in edges)
        {
            edgeIds[2 * ei + 0] = e.a;
            edgeIds[2 * ei + 1] = e.b;
            ei++;
        }

        var tris = new List<int>(faceCount.Count * 3);
        foreach (var kv in faceCount)
        {
            if (kv.Value != 1) continue;

            var key = kv.Key;
            var ori = faceOri[key];

            Vector3 p0 = vertices[ori.i0];
            Vector3 p1 = vertices[ori.i1];
            Vector3 p2 = vertices[ori.i2];
            Vector3 n = Vector3.Cross(p1 - p0, p2 - p0);

            Vector3 fc = (p0 + p1 + p2) / 3f;
            if (Vector3.Dot(n, fc) < 0f)
            {
                tris.Add(ori.i0);
                tris.Add(ori.i2);
                tris.Add(ori.i1);
            }
            else
            {
                tris.Add(ori.i0);
                tris.Add(ori.i1);
                tris.Add(ori.i2);
            }
        }

        surfaceTriIds = tris.ToArray();
    }

    private static void AddFace(
        Dictionary<FaceKey, int> faceCount,
        Dictionary<FaceKey, FaceOri> faceOri,
        int i0, int i1, int i2)
    {
        var k = new FaceKey(i0, i1, i2);
        if (faceCount.TryGetValue(k, out int c))
            faceCount[k] = c + 1;
        else
        {
            faceCount[k] = 1;
            faceOri[k] = new FaceOri(i0, i1, i2);
        }
    }
}
