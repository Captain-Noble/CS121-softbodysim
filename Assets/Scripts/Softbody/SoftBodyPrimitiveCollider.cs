// SoftBodyPrimitiveCollider.cs
// 请以 UTF-8 保存本文件
using UnityEngine;

[DisallowMultipleComponent]
public sealed class SoftBodyPrimitiveCollider : MonoBehaviour
{
    public enum PrimitiveType
    {
        Sphere = 0,
        Box = 1,
        Capsule = 2
    }

    public struct PrimitiveColliderData
    {
        public PrimitiveType type;

        // World-space center of the primitive
        public Vector3 positionW;

        // Box: OBB rotation
        // Capsule: rotation such that local Y is the capsule axis in world
        public Quaternion rotationW;

        // Sphere: x = radius
        // Box: x,y,z = halfExtents
        // Capsule: x = radius, y = halfHeight (cylinder part), axis = local Y
        public Vector3 data;
    }

    [Header("Auto from Unity Collider (recommended)")]
    [SerializeField] private bool autoFromUnityCollider = true;

    [Tooltip("可选：指定一个 Unity Collider；为空则自动 GetComponent<Collider>()")]
    [SerializeField] private Collider sourceCollider;

    [Tooltip("默认忽略 Trigger；如果你想让 Trigger 也参与推开，勾选此项")]
    [SerializeField] private bool includeTriggers = false;

    // ---- Manual fallback (only used when autoFromUnityCollider=false or no supported collider found) ----
    [Header("Manual Fallback (optional)")]
    [SerializeField] private PrimitiveType manualType = PrimitiveType.Sphere;

    [Header("Sphere")]
    [SerializeField] private float sphereRadius = 0.5f;

    [Header("Box")]
    [SerializeField] private Vector3 boxHalfExtents = Vector3.one * 0.5f;

    [Header("Capsule (local Y axis)")]
    [SerializeField] private float capsuleRadius = 0.25f;
    [SerializeField] private float capsuleHalfHeight = 0.5f;

    private Collider UnityCol
    {
        get
        {
            if (sourceCollider != null) return sourceCollider;
            sourceCollider = GetComponent<Collider>();
            return sourceCollider;
        }
    }

    public bool IsActiveForSoftBody
    {
        get
        {
            if (!isActiveAndEnabled) return false;
            var c = UnityCol;
            if (autoFromUnityCollider)
            {
                if (c == null) return false;
                if (!c.enabled) return false;
                if (!includeTriggers && c.isTrigger) return false;
            }
            return true;
        }
    }

    private void OnEnable()
    {
        // 双保险：Instance 没有也去找一个
        var mgr = SoftBodyManager.Instance != null ? SoftBodyManager.Instance : FindObjectOfType<SoftBodyManager>(true);
        if (mgr != null) mgr.RegisterPrimitiveCollider(this);
    }

    private void OnDisable()
    {
        var mgr = SoftBodyManager.Instance != null ? SoftBodyManager.Instance : FindObjectOfType<SoftBodyManager>(true);
        if (mgr != null) mgr.UnregisterPrimitiveCollider(this);
    }

    public PrimitiveColliderData GetWorldData()
    {
        if (autoFromUnityCollider)
        {
            var c = UnityCol;

            // 如果没 collider / 不支持，fallback 到 manual
            if (c is SphereCollider sc) return FromSphereCollider(sc);
            if (c is BoxCollider bc) return FromBoxCollider(bc);
            if (c is CapsuleCollider cc) return FromCapsuleCollider(cc);

            // 不支持 MeshCollider / TerrainCollider 等，自动 fallback
            return FromManualFallback();
        }

        return FromManualFallback();
    }

    private PrimitiveColliderData FromSphereCollider(SphereCollider sc)
    {
        PrimitiveColliderData d;
        d.type = PrimitiveType.Sphere;

        // Unity 的 center 是 local-space，需要 TransformPoint
        d.positionW = transform.TransformPoint(sc.center);
        d.rotationW = transform.rotation;

        float r = Mathf.Max(1e-6f, sc.radius * MaxAbsScale(transform.lossyScale));
        d.data = new Vector3(r, 0f, 0f);
        return d;
    }

    private PrimitiveColliderData FromBoxCollider(BoxCollider bc)
    {
        PrimitiveColliderData d;
        d.type = PrimitiveType.Box;

        d.positionW = transform.TransformPoint(bc.center);
        d.rotationW = transform.rotation;

        Vector3 s = AbsVec(transform.lossyScale);
        Vector3 half = 0.5f * bc.size;
        d.data = new Vector3(
            Mathf.Max(1e-6f, half.x * s.x),
            Mathf.Max(1e-6f, half.y * s.y),
            Mathf.Max(1e-6f, half.z * s.z)
        );
        return d;
    }

    private PrimitiveColliderData FromCapsuleCollider(CapsuleCollider cc)
    {
        PrimitiveColliderData d;
        d.type = PrimitiveType.Capsule;

        d.positionW = transform.TransformPoint(cc.center);

        // CapsuleCollider.direction: 0=X,1=Y,2=Z
        Vector3 axisLocal =
            (cc.direction == 0) ? Vector3.right :
            (cc.direction == 2) ? Vector3.forward :
            Vector3.up;

        // 让“我们的 capsule 数学”仍然假设轴是 local Y：
        // 通过额外旋转把 local Y 对齐到 Unity capsule 的轴
        Quaternion axisRotLocal = Quaternion.FromToRotation(Vector3.up, axisLocal);
        d.rotationW = transform.rotation * axisRotLocal;

        Vector3 s = AbsVec(transform.lossyScale);

        float sAxis = (cc.direction == 0) ? s.x : (cc.direction == 2) ? s.z : s.y;
        float sPerp1, sPerp2;
        if (cc.direction == 0) { sPerp1 = s.y; sPerp2 = s.z; }
        else if (cc.direction == 2) { sPerp1 = s.x; sPerp2 = s.y; }
        else { sPerp1 = s.x; sPerp2 = s.z; }

        // 半径用“垂直方向中最大的缩放”做保守处理（非均匀缩放下 capsule 本就不严格）
        float sRad = Mathf.Max(sPerp1, sPerp2);
        float radiusW = Mathf.Max(1e-6f, cc.radius * sRad);

        // Unity CapsuleCollider.height 包含两个半球，圆柱部分半高 = height/2 - radius
        float halfHeightW = Mathf.Max(0f, (cc.height * sAxis) * 0.5f - radiusW);

        d.data = new Vector3(radiusW, halfHeightW, 0f);
        return d;
    }

    private PrimitiveColliderData FromManualFallback()
    {
        PrimitiveColliderData d;
        d.type = manualType;
        d.positionW = transform.position;
        d.rotationW = transform.rotation;

        switch (manualType)
        {
            case PrimitiveType.Sphere:
                d.data = new Vector3(Mathf.Max(1e-6f, sphereRadius * MaxAbsScale(transform.lossyScale)), 0f, 0f);
                break;

            case PrimitiveType.Box:
                {
                    Vector3 s = AbsVec(transform.lossyScale);
                    d.data = new Vector3(
                        Mathf.Max(1e-6f, boxHalfExtents.x * s.x),
                        Mathf.Max(1e-6f, boxHalfExtents.y * s.y),
                        Mathf.Max(1e-6f, boxHalfExtents.z * s.z));
                    break;
                }

            default: // Capsule (local Y)
                {
                    float s = MaxAbsScale(transform.lossyScale);
                    d.data = new Vector3(
                        Mathf.Max(1e-6f, capsuleRadius * s),
                        Mathf.Max(0f, capsuleHalfHeight * s),
                        0f);
                    break;
                }
        }

        return d;
    }

    private static Vector3 AbsVec(Vector3 v) => new Vector3(Mathf.Abs(v.x), Mathf.Abs(v.y), Mathf.Abs(v.z));
    private static float MaxAbsScale(Vector3 s) => Mathf.Max(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
}
