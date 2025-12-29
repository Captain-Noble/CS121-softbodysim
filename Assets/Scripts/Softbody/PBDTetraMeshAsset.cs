using UnityEngine;

[CreateAssetMenu(menuName = "Softbody/PBD Tetra Mesh Asset", fileName = "PBDTetraMeshAsset")]
public sealed class PBDTetraMeshAsset : ScriptableObject
{
    public Vector3[] verticesLocal;
    public int[] edgeIds;         // length = E*2
    public int[] tetIds;          // length = T*4
    public int[] surfaceTriIds;   // for rendering only
}
