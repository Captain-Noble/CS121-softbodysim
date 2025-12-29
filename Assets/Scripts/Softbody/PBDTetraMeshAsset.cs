using UnityEngine;

[CreateAssetMenu(menuName = "PBD/PBD Tetra Mesh Asset")]
public sealed class PBDTetraMeshAsset : ScriptableObject
{
    [Header("Geometry (local space)")]
    public Vector3[] vertices;         // all tet vertices (including interior)
    public int[] tetIds;               // 4 ints per tet
    public int[] edgeIds;              // 2 ints per edge (unique)
    public int[] surfaceTriIds;        // 3 ints per triangle (surface only)

    public bool IsValid()
    {
        if (vertices == null || vertices.Length == 0) return false;
        if (tetIds == null || tetIds.Length == 0 || (tetIds.Length % 4) != 0) return false;
        if (edgeIds == null || (edgeIds.Length % 2) != 0) return false;
        if (surfaceTriIds == null || (surfaceTriIds.Length % 3) != 0) return false;
        return true;
    }
}
