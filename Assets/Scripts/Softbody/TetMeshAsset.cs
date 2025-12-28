using UnityEngine;

[CreateAssetMenu(menuName = "SoftBody/TetMeshAsset")]
public sealed class TetMeshAsset : ScriptableObject
{
    #region Data
    public Vector3[] vertices;
    public int[] tetIds;
    public int[] edgeIds;
    public int[] surfaceTriIds;
    #endregion

    #region Validation
    public bool IsValid()
    {
        if (vertices == null || vertices.Length == 0) return false;
        if (tetIds == null || tetIds.Length == 0 || (tetIds.Length & 3) != 0) return false;
        if (edgeIds == null || edgeIds.Length == 0 || (edgeIds.Length & 1) != 0) return false;
        if (surfaceTriIds == null || surfaceTriIds.Length == 0 || (surfaceTriIds.Length % 3) != 0) return false;
        return true;
    }
    #endregion
}
