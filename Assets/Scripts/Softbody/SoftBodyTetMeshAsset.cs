// SoftBodyTetMeshAsset.cs
// 请以 UTF-8 保存本文件
using Hanzzz.Tetrahedralizer;
using System;
using System.Collections.Generic;
using UnityEngine;

public sealed class SoftBodyTetMeshAsset : ScriptableObject
{
    public Vector3[] vertices;
    public int[] tetIds;         // 4 * tetCount
    public int[] edgeIds;        // 2 * edgeCount
    public int[] surfaceTriIds;  // 3 * triCount

    public bool IsValid()
    {
        if (vertices == null || vertices.Length == 0) return false;
        if (tetIds == null || tetIds.Length < 4 || (tetIds.Length & 3) != 0) return false;
        if (edgeIds == null || (edgeIds.Length & 1) != 0) return false;
        if (surfaceTriIds == null || (surfaceTriIds.Length % 3) != 0) return false;
        return true;
    }

    // runtime/editor create
    public static bool TryCreateFromMesh(Mesh src, float degenerateRatio, out SoftBodyTetMeshAsset asset)
    {
        asset = null;
        if (src == null) return false;

        var tetrahedralizer = new Tetrahedralizer();
        tetrahedralizer.SetSettings(new Tetrahedralizer.Settings(false, (double)degenerateRatio));

        var tetrahedralized = ScriptableObject.CreateInstance<TetrahedralizedMesh>();
        var tetrahedral = ScriptableObject.CreateInstance<TetrahedralMesh>();

        tetrahedralizer.MeshToTetrahedralizedMesh(src, tetrahedralized);
        tetrahedralizer.TetrahedralizedMeshToTetrahedralMesh(tetrahedralized, tetrahedral);

        if (tetrahedral.vertices == null || tetrahedral.vertices.Count == 0) return false;
        if (tetrahedral.tetrahedrons == null || tetrahedral.tetrahedrons.Count == 0) return false;
        if ((tetrahedral.tetrahedrons.Count & 3) != 0) return false;

        Vector3[] verts = tetrahedral.vertices.ToArray();
        int[] tets = tetrahedral.tetrahedrons.ToArray();

        // IMPORTANT: unify orientation (positive volume)
        OrientTetsPositive(verts, tets);

        BuildEdgesAndSurface(verts, tets, out int[] edges, out int[] surfaceTris);

        SoftBodyTetMeshAsset inst = ScriptableObject.CreateInstance<SoftBodyTetMeshAsset>();
        inst.vertices = verts;
        inst.tetIds = tets;
        inst.edgeIds = edges;
        inst.surfaceTriIds = surfaceTris;

#if UNITY_EDITOR
        string folderRoot = "Assets/SoftBody";
        string folderGen = "Assets/SoftBody/Generated";

        if (!UnityEditor.AssetDatabase.IsValidFolder(folderRoot))
            UnityEditor.AssetDatabase.CreateFolder("Assets", "SoftBody");

        if (!UnityEditor.AssetDatabase.IsValidFolder(folderGen))
            UnityEditor.AssetDatabase.CreateFolder(folderRoot, "Generated");

        string safeName = string.IsNullOrEmpty(src.name) ? "Mesh" : src.name;
        string path = UnityEditor.AssetDatabase.GenerateUniqueAssetPath($"{folderGen}/{safeName}_Tet.asset");

        UnityEditor.AssetDatabase.CreateAsset(inst, path);
        UnityEditor.AssetDatabase.SaveAssets();
        UnityEditor.AssetDatabase.Refresh();

        asset = UnityEditor.AssetDatabase.LoadAssetAtPath<SoftBodyTetMeshAsset>(path);
        return asset != null;
#else
        inst.hideFlags = HideFlags.DontSaveInBuild | HideFlags.DontSaveInEditor;
        asset = inst;
        return true;
#endif
    }

    private static void OrientTetsPositive(Vector3[] v, int[] tets)
    {
        int tetCount = tets.Length / 4;
        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            int a = tets[k + 0];
            int b = tets[k + 1];
            int c = tets[k + 2];
            int d = tets[k + 3];

            float vol = TetSignedVolume(v[a], v[b], v[c], v[d]);
            if (vol < 0f)
            {
                // swap b <-> c to flip orientation
                tets[k + 1] = c;
                tets[k + 2] = b;
            }
        }
    }

    private static float TetSignedVolume(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
    {
        Vector3 a = p1 - p0;
        Vector3 b = p2 - p0;
        Vector3 c = p3 - p0;
        return Vector3.Dot(Vector3.Cross(a, b), c) / 6f;
    }

    private readonly struct EdgeKey : IEquatable<EdgeKey>
    {
        public readonly int u, v;
        public EdgeKey(int a, int b) { if (a < b) { u = a; v = b; } else { u = b; v = a; } }
        public bool Equals(EdgeKey other) => u == other.u && v == other.v;
        public override bool Equals(object obj) => obj is EdgeKey o && Equals(o);
        public override int GetHashCode() => (u * 73856093) ^ (v * 19349663);
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
        public override bool Equals(object obj) => obj is FaceKey o && Equals(o);
        public override int GetHashCode() => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
    }

    private struct FaceVal { public int i0, i1, i2, opp, count; }

    private static void BuildEdgesAndSurface(Vector3[] verts, int[] tetIds, out int[] edgeIds, out int[] surfaceTriIds)
    {
        int tetCount = tetIds.Length / 4;

        var edgeSet = new HashSet<EdgeKey>(tetCount * 8);
        var faces = new Dictionary<FaceKey, FaceVal>(tetCount * 4);

        void AddEdge(int a, int b) => edgeSet.Add(new EdgeKey(a, b));

        void AddFace(int i0, int i1, int i2, int opp)
        {
            var key = new FaceKey(i0, i1, i2);
            if (faces.TryGetValue(key, out var v)) { v.count++; faces[key] = v; }
            else { faces[key] = new FaceVal { i0 = i0, i1 = i1, i2 = i2, opp = opp, count = 1 }; }
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
        foreach (var e in edgeSet) { edgeIds[ei++] = e.u; edgeIds[ei++] = e.v; }

        var tris = new List<int>(faces.Count * 3);

        foreach (var kv in faces)
        {
            FaceVal f = kv.Value;
            if (f.count != 1) continue; // only boundary

            int i0 = f.i0;
            int i1 = f.i1;
            int i2 = f.i2;
            int opp = f.opp;

            // ensure outward winding (opp is inside)
            Vector3 p0 = verts[i0];
            Vector3 p1 = verts[i1];
            Vector3 p2 = verts[i2];
            Vector3 po = verts[opp];

            Vector3 n = Vector3.Cross(p1 - p0, p2 - p0);
            float s = Vector3.Dot(n, po - p0);
            if (s > 0f) { int tmp = i1; i1 = i2; i2 = tmp; }

            tris.Add(i0); tris.Add(i1); tris.Add(i2);
        }

        surfaceTriIds = tris.ToArray();
    }
}
