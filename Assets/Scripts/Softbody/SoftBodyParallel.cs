// SoftBodyParallel.cs
// 请以 UTF-8 保存本文件
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using UnityEngine;

#if UNITY_2020_2_OR_NEWER
using Unity.Profiling;
#endif

public static class SoftBodyParallel
{
#if UNITY_2020_2_OR_NEWER
    private static readonly ProfilerMarker PM_EdgesGather = new ProfilerMarker("SoftBody/EdgesGather");
    private static readonly ProfilerMarker PM_EdgesApply = new ProfilerMarker("SoftBody/EdgesApply");
    private static readonly ProfilerMarker PM_TetsGather = new ProfilerMarker("SoftBody/TetsGather");
    private static readonly ProfilerMarker PM_TetsApply = new ProfilerMarker("SoftBody/TetsApply");
#endif

    // ----------------------------
    // Adjacency building (one-time)
    // ----------------------------

    // For each particle i: edgesAdjOffsets[i]..edgesAdjOffsets[i+1]-1 are incident edges.
    // At entry k:
    //   edgesAdjOther[k] = neighbor particle index
    //   edgesAdjEdge[k]  = edge constraint index
    public static void BuildEdgeAdjacency(
        int numParticles,
        int[] edgeIds,
        out int[] edgesAdjOffsets,
        out int[] edgesAdjOther,
        out int[] edgesAdjEdge)
    {
        int edgeCount = edgeIds != null ? (edgeIds.Length / 2) : 0;
        edgesAdjOffsets = new int[numParticles + 1];

        if (edgeCount == 0)
        {
            edgesAdjOther = Array.Empty<int>();
            edgesAdjEdge = Array.Empty<int>();
            return;
        }

        // count degree
        for (int e = 0; e < edgeCount; e++)
        {
            int a = edgeIds[2 * e + 0];
            int b = edgeIds[2 * e + 1];
            edgesAdjOffsets[a + 1]++;
            edgesAdjOffsets[b + 1]++;
        }

        // prefix sum -> offsets
        for (int i = 0; i < numParticles; i++)
            edgesAdjOffsets[i + 1] += edgesAdjOffsets[i];

        int total = edgesAdjOffsets[numParticles];
        edgesAdjOther = new int[total];
        edgesAdjEdge = new int[total];

        // cursor
        int[] cur = new int[numParticles];
        Array.Copy(edgesAdjOffsets, cur, numParticles);

        for (int e = 0; e < edgeCount; e++)
        {
            int a = edgeIds[2 * e + 0];
            int b = edgeIds[2 * e + 1];

            int ka = cur[a]++;
            edgesAdjOther[ka] = b;
            edgesAdjEdge[ka] = e;

            int kb = cur[b]++;
            edgesAdjOther[kb] = a;
            edgesAdjEdge[kb] = e;
        }
    }

    // For each particle i: tetsAdjOffsets[i]..tetsAdjOffsets[i+1]-1 are incident tets.
    // At entry k:
    //   tetsAdjTet[k]  = tet constraint index
    //   tetsAdjRole[k] = 0..3 (which vertex of the tet this particle is)
    public static void BuildTetAdjacency(
        int numParticles,
        int[] tetIds,
        out int[] tetsAdjOffsets,
        out int[] tetsAdjTet,
        out byte[] tetsAdjRole)
    {
        int tetCount = tetIds != null ? (tetIds.Length / 4) : 0;
        tetsAdjOffsets = new int[numParticles + 1];

        if (tetCount == 0)
        {
            tetsAdjTet = Array.Empty<int>();
            tetsAdjRole = Array.Empty<byte>();
            return;
        }

        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            int a = tetIds[k + 0];
            int b = tetIds[k + 1];
            int c = tetIds[k + 2];
            int d = tetIds[k + 3];
            tetsAdjOffsets[a + 1]++;
            tetsAdjOffsets[b + 1]++;
            tetsAdjOffsets[c + 1]++;
            tetsAdjOffsets[d + 1]++;
        }

        for (int i = 0; i < numParticles; i++)
            tetsAdjOffsets[i + 1] += tetsAdjOffsets[i];

        int total = tetsAdjOffsets[numParticles];
        tetsAdjTet = new int[total];
        tetsAdjRole = new byte[total];

        int[] cur = new int[numParticles];
        Array.Copy(tetsAdjOffsets, cur, numParticles);

        for (int t = 0; t < tetCount; t++)
        {
            int k = 4 * t;
            int a = tetIds[k + 0];
            int b = tetIds[k + 1];
            int c = tetIds[k + 2];
            int d = tetIds[k + 3];

            int ka = cur[a]++; tetsAdjTet[ka] = t; tetsAdjRole[ka] = 0;
            int kb = cur[b]++; tetsAdjTet[kb] = t; tetsAdjRole[kb] = 1;
            int kc = cur[c]++; tetsAdjTet[kc] = t; tetsAdjRole[kc] = 2;
            int kd = cur[d]++; tetsAdjTet[kd] = t; tetsAdjRole[kd] = 3;
        }
    }

    // ----------------------------
    // Pre / Post (unchanged)
    // ----------------------------

    public static void PreSolveSerial(
        Vector3[] posL,
        Vector3[] prevPosL,
        Vector3[] velL,
        float[] invMass,
        float dt,
        Vector3 gravityL,
        bool groundCollision,
        float groundY,
        Matrix4x4 localToWorld,
        Matrix4x4 worldToLocal)
    {
        int n = posL.Length;
        for (int i = 0; i < n; i++)
        {
            if (invMass[i] == 0f) continue;

            velL[i] += gravityL * dt;
            prevPosL[i] = posL[i];
            posL[i] += velL[i] * dt;

            if (groundCollision)
            {
                Vector3 pw = localToWorld.MultiplyPoint3x4(posL[i]);
                if (pw.y < groundY)
                {
                    pw.y = groundY;
                    posL[i] = worldToLocal.MultiplyPoint3x4(pw);
                }
            }
        }
    }

    public static void PreSolveParallel(
        Vector3[] posL,
        Vector3[] prevPosL,
        Vector3[] velL,
        float[] invMass,
        float dt,
        Vector3 gravityL,
        bool groundCollision,
        float groundY,
        Matrix4x4 localToWorld,
        Matrix4x4 worldToLocal,
        ParallelOptions opt)
    {
        int n = posL.Length;
        Parallel.ForEach(Partitioner.Create(0, n, 1024), opt, range =>
        {
            for (int i = range.Item1; i < range.Item2; i++)
            {
                if (invMass[i] == 0f) continue;

                velL[i] += gravityL * dt;
                prevPosL[i] = posL[i];
                posL[i] += velL[i] * dt;

                if (groundCollision)
                {
                    Vector3 pw = localToWorld.MultiplyPoint3x4(posL[i]);
                    if (pw.y < groundY)
                    {
                        pw.y = groundY;
                        posL[i] = worldToLocal.MultiplyPoint3x4(pw);
                    }
                }
            }
        });
    }

    public static void PostSolveSerial(
        Vector3[] posL,
        Vector3[] prevPosL,
        Vector3[] velL,
        float[] invMass,
        float dt,
        Vector3[] renderVerts)
    {
        float invDt = dt > 0f ? 1f / dt : 0f;
        int n = posL.Length;

        for (int i = 0; i < n; i++)
        {
            if (invMass[i] == 0f) continue;
            velL[i] = (posL[i] - prevPosL[i]) * invDt;
        }

        Array.Copy(posL, renderVerts, n);
    }

    public static void PostSolveParallel(
        Vector3[] posL,
        Vector3[] prevPosL,
        Vector3[] velL,
        float[] invMass,
        float dt,
        Vector3[] renderVerts,
        ParallelOptions opt)
    {
        float invDt = dt > 0f ? 1f / dt : 0f;
        int n = posL.Length;

        Parallel.ForEach(Partitioner.Create(0, n, 1024), opt, range =>
        {
            for (int i = range.Item1; i < range.Item2; i++)
            {
                if (invMass[i] == 0f) continue;
                velL[i] = (posL[i] - prevPosL[i]) * invDt;
            }
        });

        Array.Copy(posL, renderVerts, n);
    }

    // ----------------------------
    // NVIDIA-style parallel solve:
    // Jacobi gather + averaging + SOR omega
    // ----------------------------

    public static void SolveEdgesJacobiAveraged(
        Vector3[] posL,
        float[] invMass,
        int[] edgeIds,
        float[] restEdgeLen,
        int[] edgesAdjOffsets,
        int[] edgesAdjOther,
        int[] edgesAdjEdge,
        float compliance,
        float dt,
        float omega,
        Vector3[] deltaOut,
        int[] countOut,
        ParallelOptions opt)
    {
        int n = posL.Length;
        if (edgeIds == null || restEdgeLen == null || restEdgeLen.Length == 0) return;

        float alpha = dt > 0f ? (compliance / (dt * dt)) : 0f;

#if UNITY_2020_2_OR_NEWER
        using (PM_EdgesGather.Auto())
#endif
        {
            Parallel.ForEach(Partitioner.Create(0, n, 1024), opt, range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    if (invMass[i] == 0f) { deltaOut[i] = Vector3.zero; countOut[i] = 0; continue; }

                    int begin = edgesAdjOffsets[i];
                    int end = edgesAdjOffsets[i + 1];

                    Vector3 xi = posL[i];
                    float wi = invMass[i];

                    Vector3 sum = Vector3.zero;
                    int cnt = 0;

                    for (int k = begin; k < end; k++)
                    {
                        int j = edgesAdjOther[k];
                        float wj = invMass[j];
                        if (wj == 0f && wi == 0f) continue;

                        int e = edgesAdjEdge[k];

                        Vector3 xj = posL[j];
                        Vector3 d = xi - xj;
                        float len2 = d.sqrMagnitude;
                        if (len2 < 1e-18f) continue;

                        float len = Mathf.Sqrt(len2);
                        float C = len - restEdgeLen[e];

                        float w = wi + wj;
                        if (w == 0f) continue;

                        // XPBD-like compliance in denominator (no persistent lambda here)
                        float s = -C / (w + alpha);

                        Vector3 nrm = d / len;
                        // particle-centric gather: only compute my own delta
                        sum += nrm * (s * wi);
                        cnt++;
                    }

                    deltaOut[i] = sum;
                    countOut[i] = cnt;
                }
            });
        }

#if UNITY_2020_2_OR_NEWER
        using (PM_EdgesApply.Auto())
#endif
        {
            Parallel.ForEach(Partitioner.Create(0, n, 1024), opt, range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    int cnt = countOut[i];
                    if (cnt <= 0) continue;
                    if (invMass[i] == 0f) continue;

                    posL[i] += (omega / cnt) * deltaOut[i];
                }
            });
        }
    }

    public static void SolveVolumesJacobiAveraged(
        Vector3[] posL,
        float[] invMass,
        int[] tetIds,
        float[] restVol,
        int[] tetsAdjOffsets,
        int[] tetsAdjTet,
        byte[] tetsAdjRole,
        float compliance,
        float dt,
        float omega,
        Vector3[] deltaOut,
        int[] countOut,
        ParallelOptions opt)
    {
        int n = posL.Length;
        if (tetIds == null || restVol == null || restVol.Length == 0) return;

        float alpha = dt > 0f ? (compliance / (dt * dt)) : 0f;

#if UNITY_2020_2_OR_NEWER
        using (PM_TetsGather.Auto())
#endif
        {
            Parallel.ForEach(Partitioner.Create(0, n, 512), opt, range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    if (invMass[i] == 0f) { deltaOut[i] = Vector3.zero; countOut[i] = 0; continue; }

                    int begin = tetsAdjOffsets[i];
                    int end = tetsAdjOffsets[i + 1];

                    Vector3 sum = Vector3.zero;
                    int cnt = 0;

                    for (int kk = begin; kk < end; kk++)
                    {
                        int t = tetsAdjTet[kk];
                        int role = tetsAdjRole[kk];

                        int baseId = 4 * t;
                        int a = tetIds[baseId + 0];
                        int b = tetIds[baseId + 1];
                        int c = tetIds[baseId + 2];
                        int d = tetIds[baseId + 3];

                        float wa = invMass[a];
                        float wb = invMass[b];
                        float wc = invMass[c];
                        float wd = invMass[d];
                        if (wa + wb + wc + wd == 0f) continue;

                        Vector3 pa = posL[a];
                        Vector3 pb = posL[b];
                        Vector3 pc = posL[c];
                        Vector3 pd = posL[d];

                        Vector3 ga = Vector3.Cross(pd - pb, pc - pb) / 6f;
                        Vector3 gb = Vector3.Cross(pc - pa, pd - pa) / 6f;
                        Vector3 gc = Vector3.Cross(pd - pa, pb - pa) / 6f;
                        Vector3 gd = Vector3.Cross(pb - pa, pc - pa) / 6f;

                        float wsum =
                            wa * Vector3.Dot(ga, ga) +
                            wb * Vector3.Dot(gb, gb) +
                            wc * Vector3.Dot(gc, gc) +
                            wd * Vector3.Dot(gd, gd);

                        if (wsum == 0f) continue;

                        float vol = TetVolume(pa, pb, pc, pd);
                        float C = vol - restVol[t];

                        float s = -C / (wsum + alpha);

                        // choose my gradient + weight
                        Vector3 g;
                        float wi;
                        switch (role)
                        {
                            case 0: g = ga; wi = wa; break;
                            case 1: g = gb; wi = wb; break;
                            case 2: g = gc; wi = wc; break;
                            default: g = gd; wi = wd; break;
                        }

                        if (wi == 0f) continue;

                        sum += g * (s * wi);
                        cnt++;
                    }

                    deltaOut[i] = sum;
                    countOut[i] = cnt;
                }
            });
        }

#if UNITY_2020_2_OR_NEWER
        using (PM_TetsApply.Auto())
#endif
        {
            Parallel.ForEach(Partitioner.Create(0, n, 512), opt, range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    int cnt = countOut[i];
                    if (cnt <= 0) continue;
                    if (invMass[i] == 0f) continue;

                    posL[i] += (omega / cnt) * deltaOut[i];
                }
            });
        }
    }

    // ----------------------------
    // Ground (unchanged)
    // ----------------------------
    public static void SolveGroundSerial(
        Vector3[] posL,
        float[] invMass,
        float groundY,
        Matrix4x4 localToWorld,
        Matrix4x4 worldToLocal)
    {
        int n = posL.Length;
        for (int i = 0; i < n; i++)
        {
            if (invMass[i] == 0f) continue;
            Vector3 pw = localToWorld.MultiplyPoint3x4(posL[i]);
            if (pw.y < groundY)
            {
                pw.y = groundY;
                posL[i] = worldToLocal.MultiplyPoint3x4(pw);
            }
        }
    }

    // ----------------------------
    // Math (unchanged)
    // ----------------------------
    public static float TetVolume(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3)
    {
        Vector3 a = p1 - p0;
        Vector3 b = p2 - p0;
        Vector3 c = p3 - p0;
        return Vector3.Dot(Vector3.Cross(a, b), c) / 6f;
    }
}
