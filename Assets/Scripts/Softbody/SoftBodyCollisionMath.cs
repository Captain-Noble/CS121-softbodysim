// SoftBodyCollisionMath.cs
// 请以 UTF-8 保存本文件
using UnityEngine;

public static class SoftBodyCollisionMath
{
    // return true if penetration, pushW is the minimal translation to resolve (world-space).
    public static bool ComputePushOut(in SoftBodyPrimitiveCollider.PrimitiveColliderData col, Vector3 pointW, float particleRadius, out Vector3 pushW)
    {
        switch (col.type)
        {
            case SoftBodyPrimitiveCollider.PrimitiveType.Sphere:
                return PushOutSphere(col.positionW, col.data.x + particleRadius, pointW, out pushW);

            case SoftBodyPrimitiveCollider.PrimitiveType.Box:
                return PushOutBox(col.positionW, col.rotationW, col.data, particleRadius, pointW, out pushW);

            default: // Capsule
                return PushOutCapsule(col.positionW, col.rotationW, col.data.x, col.data.y, particleRadius, pointW, out pushW);
        }
    }

    private static bool PushOutSphere(Vector3 center, float radius, Vector3 p, out Vector3 push)
    {
        Vector3 v = p - center;
        float d2 = v.sqrMagnitude;
        float r = Mathf.Max(1e-6f, radius);

        if (d2 >= r * r)
        {
            push = Vector3.zero;
            return false;
        }

        float d = Mathf.Sqrt(Mathf.Max(d2, 1e-20f));
        Vector3 n = (d > 1e-10f) ? (v / d) : Vector3.up;
        push = n * (r - d);
        return true;
    }

    // Oriented box (OBB). data = halfExtents.
    // 只在“点在盒子内部（含padding）”时返回 true 并 push-out。
    private static bool PushOutBox(Vector3 center, Quaternion rot, Vector3 halfExt, float particleRadius, Vector3 pW, out Vector3 pushW)
    {
        Quaternion inv = Quaternion.Inverse(rot);
        Vector3 pL = inv * (pW - center);

        Vector3 e = halfExt + Vector3.one * particleRadius;

        bool inside =
            Mathf.Abs(pL.x) <= e.x &&
            Mathf.Abs(pL.y) <= e.y &&
            Mathf.Abs(pL.z) <= e.z;

        if (!inside)
        {
            pushW = Vector3.zero;
            return false;
        }

        // inside: push out along minimum distance to a face
        float dx = e.x - Mathf.Abs(pL.x);
        float dy = e.y - Mathf.Abs(pL.y);
        float dz = e.z - Mathf.Abs(pL.z);

        if (dx <= dy && dx <= dz)
        {
            float sx = pL.x >= 0f ? 1f : -1f;
            Vector3 pushL = new Vector3(dx * sx, 0f, 0f);
            pushW = rot * pushL;
            return true;
        }
        if (dy <= dz)
        {
            float sy = pL.y >= 0f ? 1f : -1f;
            Vector3 pushL = new Vector3(0f, dy * sy, 0f);
            pushW = rot * pushL;
            return true;
        }
        else
        {
            float sz = pL.z >= 0f ? 1f : -1f;
            Vector3 pushL = new Vector3(0f, 0f, dz * sz);
            pushW = rot * pushL;
            return true;
        }
    }

    // Capsule axis = local Y in collider space. data: radius, halfHeight (cylinder half height).
    private static bool PushOutCapsule(Vector3 center, Quaternion rot, float radius, float halfHeight, float particleRadius, Vector3 pW, out Vector3 pushW)
    {
        float r = Mathf.Max(1e-6f, radius + particleRadius);
        float h = Mathf.Max(0f, halfHeight);

        Vector3 up = rot * Vector3.up;
        Vector3 a = center - up * h;
        Vector3 b = center + up * h;

        Vector3 ab = b - a;
        float t = 0f;
        float ab2 = ab.sqrMagnitude;
        if (ab2 > 1e-20f)
        {
            t = Vector3.Dot(pW - a, ab) / ab2;
            t = Mathf.Clamp01(t);
        }
        Vector3 c = a + ab * t;

        return PushOutSphere(c, r, pW, out pushW);
    }
}
