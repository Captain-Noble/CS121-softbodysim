using UnityEngine;

public sealed class RuntimeTetMesh
{
    public Vector3[] vertices;     // tet vertices (unique positions)
    public int[] tetIds;           // 4 ints per tet (scheme 1)
    public int[] edgeIds;          // 2 ints per edge (unique)
    public int[] surfaceTriIds;    // 3 ints per tri (extracted from tets)

    public bool IsValid =>
        vertices != null && vertices.Length > 0 &&
        tetIds != null && tetIds.Length > 0 && (tetIds.Length % 4) == 0 &&
        edgeIds != null && (edgeIds.Length % 2) == 0 &&
        surfaceTriIds != null && (surfaceTriIds.Length % 3) == 0;
}
