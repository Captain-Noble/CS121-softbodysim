// PBDRemoteSoftBody.cs
using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public sealed class PBDRemoteSoftBody : MonoBehaviour
{
    [Header("Auto Tetrahedralize")]
    public bool autoTetrahedralize = true;

    [Range(0.0f, 1.0f)]
    public double degenerateTetrahedronRatio = 0.0;

    [Header("Pinning")]
    public bool pinTopLayer = false;
    public float pinTopEpsilon = 1e-4f;

    [Header("Rendering")]
    public bool updateNormals = false;
    public int normalsEveryNFrames = 0;

    private Vector3[] _verticesLocal;
    private int[] _tetIds;
    private int[] _edgeIds;
    private int[] _surfaceTriIds;

    private Mesh _mesh;
    private Vector3[] _meshVerts;
    private int[] _meshTris;
    private int _frameCounter;

    private float[] _initWorldPos;
    private int[] _pinnedIndices;

    public int VertexCount => _verticesLocal != null ? _verticesLocal.Length : 0;
    public int EdgeCount => _edgeIds != null ? _edgeIds.Length / 2 : 0;
    public int TetCount => _tetIds != null ? _tetIds.Length / 4 : 0;

    public Vector3[] VerticesLocal => _verticesLocal;
    public int[] EdgeIds => _edgeIds;
    public int[] TetIds => _tetIds;

    public float[] InitWorldPositionsBuffer => _initWorldPos;
    public int[] PinnedIndicesBuffer => _pinnedIndices;

    private void OnEnable()
    {
        if (autoTetrahedralize)
        {
            if (!TryBuildFromMeshFilter(out var err))
            {
                Debug.LogError($"[{name}] Auto tetrahedralize failed: {err}", this);
                enabled = false;
                return;
            }
        }

        if (_verticesLocal == null || _tetIds == null || _edgeIds == null || _surfaceTriIds == null)
        {
            Debug.LogError($"[{name}] Missing tetra data.", this);
            enabled = false;
            return;
        }

        BuildRenderMesh();
        CachePinnedIndicesMainThread();
        CaptureInitWorldPositionsMainThread();

        var world = FindAnyObjectByType<PBDRemoteWorld>();
        if (world == null)
        {
            Debug.LogError("No PBDRemoteWorld found.");
            enabled = false;
            return;
        }
        world.BindBody(this);
    }

    private bool TryBuildFromMeshFilter(out string error)
    {
        error = null;

        var mf = GetComponent<MeshFilter>();
        var src = mf != null ? mf.sharedMesh : null;
        if (src == null)
        {
            error = "MeshFilter.sharedMesh is null.";
            return false;
        }

        if (!TetrahedralizerRuntimeBridge.IsAvailable())
        {
            error = "Tetrahedralizer plugin not detected (types not found at runtime).";
            return false;
        }

        if (!TetrahedralizerRuntimeBridge.TryTetrahedralizeMesh(
                src,
                remapVertexData: false,
                degenerateTetrahedronRatio: degenerateTetrahedronRatio,
                out var tet,
                out error))
        {
            return false;
        }

        _verticesLocal = tet.vertices;
        _tetIds = tet.tetrahedrons;

        BuildEdgesAndSurface(_verticesLocal, _tetIds, out _edgeIds, out _surfaceTriIds);
        return true;
    }

    private void BuildRenderMesh()
    {
        _mesh = new Mesh();
        _mesh.name = $"{name}_RemoteTetSurface";
        _mesh.indexFormat = _verticesLocal.Length > 65535
            ? UnityEngine.Rendering.IndexFormat.UInt32
            : UnityEngine.Rendering.IndexFormat.UInt16;

        _meshVerts = new Vector3[_verticesLocal.Length];
        Array.Copy(_verticesLocal, _meshVerts, _meshVerts.Length);

        _meshTris = _surfaceTriIds;

        _mesh.vertices = _meshVerts;
        _mesh.triangles = _meshTris;

        _mesh.RecalculateBounds();
        if (updateNormals) _mesh.RecalculateNormals();
        _mesh.MarkDynamic();

        GetComponent<MeshFilter>().sharedMesh = _mesh;
    }

    private void CaptureInitWorldPositionsMainThread()
    {
        int n = VertexCount;
        if (n <= 0)
        {
            _initWorldPos = Array.Empty<float>();
            return;
        }

        int need = n * 3;
        if (_initWorldPos == null || _initWorldPos.Length != need)
            _initWorldPos = new float[need];

        var l2w = transform.localToWorldMatrix;
        for (int i = 0; i < n; i++)
        {
            Vector3 p = l2w.MultiplyPoint3x4(_verticesLocal[i]);
            int k = i * 3;
            _initWorldPos[k + 0] = p.x;
            _initWorldPos[k + 1] = p.y;
            _initWorldPos[k + 2] = p.z;
        }
    }

    private void CachePinnedIndicesMainThread()
    {
        if (!pinTopLayer || _verticesLocal == null || _verticesLocal.Length == 0)
        {
            _pinnedIndices = Array.Empty<int>();
            return;
        }

        float topY = float.NegativeInfinity;
        for (int i = 0; i < _verticesLocal.Length; i++)
            if (_verticesLocal[i].y > topY) topY = _verticesLocal[i].y;

        var list = new List<int>(128);
        for (int i = 0; i < _verticesLocal.Length; i++)
        {
            if (Mathf.Abs(_verticesLocal[i].y - topY) <= pinTopEpsilon)
                list.Add(i);
        }

        _pinnedIndices = list.Count == 0 ? Array.Empty<int>() : list.ToArray();
    }

    public void ApplyWorldPositions(float[] worldPos)
    {
        if (_mesh == null || _meshVerts == null) return;

        int n = VertexCount;
        if (worldPos == null || worldPos.Length < n * 3) return;

        var w2l = transform.worldToLocalMatrix;
        for (int i = 0; i < n; i++)
        {
            int k = i * 3;
            Vector3 p = new Vector3(worldPos[k + 0], worldPos[k + 1], worldPos[k + 2]);
            _meshVerts[i] = w2l.MultiplyPoint3x4(p);
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

    private readonly struct EdgeKey : IEquatable<EdgeKey>
    {
        public readonly int a, b;
        public EdgeKey(int i0, int i1)
        {
            if (i0 < i1) { a = i0; b = i1; }
            else { a = i1; b = i0; }
        }
        public bool Equals(EdgeKey other) => a == other.a && b == other.b;
        public override int GetHashCode() => (a * 73856093) ^ (b * 19349663);
    }

    private readonly struct FaceKey : IEquatable<FaceKey>
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

    private readonly struct FaceOri
    {
        public readonly int i0, i1, i2;
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
