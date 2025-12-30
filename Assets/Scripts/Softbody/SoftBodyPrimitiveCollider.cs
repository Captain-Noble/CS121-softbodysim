// SoftBodyPrimitiveCollider.cs
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
        public Vector3 positionW;
        public Quaternion rotationW;
        public Vector3 data;
    }

    [Header("Auto from Unity Collider")]
    [SerializeField] private bool autoFromUnityCollider = true;
    [SerializeField] private Collider sourceCollider;
    [SerializeField] private bool includeTriggers = false;

    [Header("Manual Fallback")]
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

            if (autoFromUnityCollider)
            {
                var c = UnityCol;
                if (c == null) return false;
                if (!c.enabled) return false;
                if (!includeTriggers && c.isTrigger) return false;
            }

            return true;
        }
    }

    private void OnEnable()
    {
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

            if (c is SphereCollider sc) return FromSphereCollider(sc);
            if (c is BoxCollider bc) return FromBoxCollider(bc);
            if (c is CapsuleCollider cc) return FromCapsuleCollider(cc);

            return FromManualFallback();
        }

        return FromManualFallback();
    }

    private PrimitiveColliderData FromSphereCollider(SphereCollider sc)
    {
        PrimitiveColliderData d;
        d.type = PrimitiveType.Sphere;
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

        Vector3 axisLocal =
            (cc.direction == 0) ? Vector3.right :
            (cc.direction == 2) ? Vector3.forward :
            Vector3.up;

        Quaternion axisRotLocal = Quaternion.FromToRotation(Vector3.up, axisLocal);
        d.rotationW = transform.rotation * axisRotLocal;

        Vector3 s = AbsVec(transform.lossyScale);

        float sAxis = (cc.direction == 0) ? s.x : (cc.direction == 2) ? s.z : s.y;
        float sPerp1, sPerp2;
        if (cc.direction == 0) { sPerp1 = s.y; sPerp2 = s.z; }
        else if (cc.direction == 2) { sPerp1 = s.x; sPerp2 = s.y; }
        else { sPerp1 = s.x; sPerp2 = s.z; }

        float sRad = Mathf.Max(sPerp1, sPerp2);
        float radiusW = Mathf.Max(1e-6f, cc.radius * sRad);
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

            default:
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
