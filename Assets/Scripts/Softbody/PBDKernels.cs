using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public static class PBDKernels
{
    public sealed class LocalAccum
    {
        public readonly Vector3[] delta;
        public readonly int[] count;

        private int[] _touched;
        private int _touchedCount;

        public LocalAccum(int n, int touchedCapacity)
        {
            delta = new Vector3[n];
            count = new int[n];
            _touched = new int[Mathf.Max(1024, touchedCapacity)];
            _touchedCount = 0;
        }

        public void ResetTouched()
        {
            for (int k = 0; k < _touchedCount; k++)
            {
                int i = _touched[k];
                delta[i] = Vector3.zero;
                count[i] = 0;
            }
            _touchedCount = 0;
        }

        public void Add(int i, Vector3 d)
        {
            if (count[i] == 0)
            {
                if (_touchedCount >= _touched.Length) Array.Resize(ref _touched, _touched.Length * 2);
                _touched[_touchedCount++] = i;
            }
            delta[i] += d;
            count[i] += 1;
        }

        public int TouchedCount => _touchedCount;
        public int GetTouched(int k) => _touched[k];
    }

    public static int EstimateTouchedCapacityEdges(int particleCount, int edgeCount, int threads)
    {
        threads = Mathf.Max(1, threads);
        long approx = (long)edgeCount * 2L / threads;
        approx = (long)(approx * 2.2f);
        approx = Math.Clamp(approx, 1024, particleCount);
        return (int)approx;
    }

    public static int EstimateTouchedCapacityTets(int particleCount, int tetCount, int threads)
    {
        threads = Mathf.Max(1, threads);
        long approx = (long)tetCount * 4L / threads;
        approx = (long)(approx * 2.2f);
        approx = Math.Clamp(approx, 1024, particleCount);
        return (int)approx;
    }

    public static void Predict(Vector3[] x, Vector3[] v, float[] w, Vector3[] xStar, Vector3 gravity, float dt, bool parallel, ParallelOptions opt)
    {
        int n = x.Length;
        if (!parallel)
        {
            for (int i = 0; i < n; i++)
            {
                if (w[i] == 0f) { xStar[i] = x[i]; continue; }
                v[i] += gravity * dt;
                xStar[i] = x[i] + v[i] * dt;
            }
            return;
        }

        Parallel.For(0, n, opt, i =>
        {
            if (w[i] == 0f) { xStar[i] = x[i]; return; }
            v[i] += gravity * dt;
            xStar[i] = x[i] + v[i] * dt;
        });
    }

    public static void Commit(
        Vector3[] x, Vector3[] v, float[] w, Vector3[] xStar, float invDt,
        bool groundEnabled, float groundY, float groundContactEps, float groundTangentDamping,
        bool parallel, ParallelOptions opt)
    {
        int n = x.Length;
        float dt = invDt > 1e-12f ? 1f / invDt : 0f;
        float damp = Mathf.Max(0f, groundTangentDamping);
        float eps = Mathf.Max(0f, groundContactEps);
        float y0 = groundY;
        float dampFactor = (damp > 0f && dt > 0f) ? Mathf.Exp(-damp * dt) : 1f;

        if (!parallel)
        {
            for (int i = 0; i < n; i++)
            {
                if (w[i] == 0f) { v[i] = Vector3.zero; xStar[i] = x[i]; continue; }

                Vector3 newX = xStar[i];
                Vector3 oldX = x[i];
                Vector3 vNew = (newX - oldX) * invDt;

                if (groundEnabled && damp > 0f && newX.y <= y0 + 2f * eps)
                {
                    vNew.x *= dampFactor;
                    vNew.z *= dampFactor;
                }

                v[i] = vNew;
                x[i] = newX;
            }
            return;
        }

        Parallel.For(0, n, opt, i =>
        {
            if (w[i] == 0f) { v[i] = Vector3.zero; xStar[i] = x[i]; return; }

            Vector3 newX = xStar[i];
            Vector3 oldX = x[i];
            Vector3 vNew = (newX - oldX) * invDt;

            if (groundEnabled && damp > 0f && newX.y <= y0 + 2f * eps)
            {
                vNew.x *= dampFactor;
                vNew.z *= dampFactor;
            }

            v[i] = vNew;
            x[i] = newX;
        });
    }

    public static void ResetMerged(Vector3[] mergedDelta, int[] mergedCount, int[] mergedTouched, ref int mergedTouchedCount)
    {
        if (mergedTouchedCount < 0) mergedTouchedCount = 0;
        if (mergedTouchedCount > mergedTouched.Length) mergedTouchedCount = mergedTouched.Length;

        for (int k = 0; k < mergedTouchedCount; k++)
        {
            int i = mergedTouched[k];
            mergedDelta[i] = Vector3.zero;
            mergedCount[i] = 0;
        }
        mergedTouchedCount = 0;
    }

    public static void ApplyMergedDelta(
        Vector3[] xStar,
        Vector3[] mergedDelta, int[] mergedCount,
        int[] mergedTouched, int mergedTouchedCount,
        float omega, bool parallel, ParallelOptions opt)
    {
        if (mergedTouchedCount <= 0) return;

        if (!parallel)
        {
            for (int kk = 0; kk < mergedTouchedCount; kk++)
            {
                int i = mergedTouched[kk];
                int c = mergedCount[i];
                if (c > 0) xStar[i] += omega * (mergedDelta[i] / c);
            }
            return;
        }

        Parallel.For(0, mergedTouchedCount, opt, kk =>
        {
            int i = mergedTouched[kk];
            int c = mergedCount[i];
            if (c > 0) xStar[i] += omega * (mergedDelta[i] / c);
        });
    }

    public static void SolveEdgesComputeParallel(
        int[] edgeI0, int[] edgeI1, float[] edgeRest, float[] edgeCompliance, float[] edgeLambda,
        Vector3[] xStar, float[] w, float dt,
        ThreadLocal<LocalAccum> tls, ParallelOptions opt)
    {
        float invDt2 = dt > 1e-12f ? 1f / (dt * dt) : 0f;
        int m = edgeI0.Length;

        Parallel.For(0, m, opt, ei =>
        {
            var acc = tls.Value;

            int i0 = edgeI0[ei];
            int i1 = edgeI1[ei];
            float w0 = w[i0], w1 = w[i1];
            float wSum = w0 + w1;
            if (wSum == 0f) return;

            Vector3 p0 = xStar[i0];
            Vector3 p1 = xStar[i1];

            Vector3 d = p0 - p1;
            float len = d.magnitude;
            if (len < 1e-12f) return;

            float C = len - edgeRest[ei];

            float alpha = edgeCompliance[ei] * invDt2;
            float lambda = edgeLambda[ei];
            float dlambda = (-C - alpha * lambda) / (wSum + alpha);
            edgeLambda[ei] = lambda + dlambda;

            Vector3 corr = (dlambda / len) * d;

            acc.Add(i0, w0 * corr);
            acc.Add(i1, -w1 * corr);
        });
    }

    public static void SolveTetsComputeParallel(
        int[] tetA, int[] tetB, int[] tetC, int[] tetD, float[] tetRestVol, float[] tetCompliance, float[] tetLambda,
        Vector3[] xStar, float[] w, float dt,
        ThreadLocal<LocalAccum> tls, ParallelOptions opt)
    {
        float invDt2 = dt > 1e-12f ? 1f / (dt * dt) : 0f;
        int m = tetA.Length;

        Parallel.For(0, m, opt, ti =>
        {
            var acc = tls.Value;

            int a = tetA[ti], b = tetB[ti], c = tetC[ti], d = tetD[ti];
            float wa = w[a], wb = w[b], wc = w[c], wd = w[d];
            if (wa + wb + wc + wd == 0f) return;

            Vector3 pa = xStar[a];
            Vector3 pb = xStar[b];
            Vector3 pc = xStar[c];
            Vector3 pd = xStar[d];

            Vector3 ga = Vector3.Cross(pd - pb, pc - pb) / 6f;
            Vector3 gb = Vector3.Cross(pc - pa, pd - pa) / 6f;
            Vector3 gc = Vector3.Cross(pd - pa, pb - pa) / 6f;
            Vector3 gd = Vector3.Cross(pb - pa, pc - pa) / 6f;

            float wSum =
                wa * Vector3.Dot(ga, ga) +
                wb * Vector3.Dot(gb, gb) +
                wc * Vector3.Dot(gc, gc) +
                wd * Vector3.Dot(gd, gd);

            if (wSum < 1e-20f) return;

            float vol = TetVolume(pa, pb, pc, pd);
            float C = vol - tetRestVol[ti];

            float alpha = tetCompliance[ti] * invDt2;
            float lambda = tetLambda[ti];
            float dlambda = (-C - alpha * lambda) / (wSum + alpha);
            tetLambda[ti] = lambda + dlambda;

            acc.Add(a, wa * (dlambda * ga));
            acc.Add(b, wb * (dlambda * gb));
            acc.Add(c, wc * (dlambda * gc));
            acc.Add(d, wd * (dlambda * gd));
        });
    }

    public static void MergeRegisteredAccums(
        LocalAccum[] accums, int count,
        Vector3[] mergedDelta, int[] mergedCount,
        int[] mergedTouched, ref int mergedTouchedCount)
    {
        for (int ai = 0; ai < count; ai++)
        {
            var acc = accums[ai];
            int tc = acc.TouchedCount;
            for (int k = 0; k < tc; k++)
            {
                int i = acc.GetTouched(k);

                if (mergedCount[i] == 0)
                {
                    mergedTouched[mergedTouchedCount++] = i;
                }

                mergedDelta[i] += acc.delta[i];
                mergedCount[i] += acc.count[i];
            }
        }
    }

    public static void ResetRegisteredAccums(LocalAccum[] accums, int count)
    {
        for (int i = 0; i < count; i++) accums[i].ResetTouched();
    }

    public static void ProjectGroundFriction(
        Vector3[] x, Vector3[] xStar, float[] w,
        float groundY, bool frictionEnabled, float muS, float muD, float contactEps, float slop,
        bool parallel, ParallelOptions opt)
    {
        int n = xStar.Length;
        float y0 = groundY;
        Vector3 N = Vector3.up;

        muS = Mathf.Max(0f, muS);
        muD = Mathf.Max(0f, muD);
        contactEps = Mathf.Max(0f, contactEps);
        slop = Mathf.Max(0f, slop);

        if (!parallel)
        {
            for (int i = 0; i < n; i++)
            {
                if (w[i] == 0f) continue;
                Vector3 p = xStar[i];
                if (p.y > y0 + contactEps) continue;

                float normalPush;
                if (p.y < y0) { normalPush = y0 - p.y; p.y = y0; }
                else normalPush = contactEps;

                if (frictionEnabled)
                {
                    Vector3 dx = p - x[i];
                    float dxN = Vector3.Dot(dx, N);
                    Vector3 dxT = dx - dxN * N;
                    float tLen = dxT.magnitude;
                    if (tLen > 1e-12f)
                    {
                        float staticLimit = muS * normalPush;
                        if (tLen <= staticLimit) p -= dxT;
                        else
                        {
                            float dynReduce = muD * normalPush;
                            float k = Mathf.Min(1f, dynReduce / tLen);
                            p -= dxT * k;
                        }
                    }
                }

                p.y = y0 + slop;
                xStar[i] = p;
            }
            return;
        }

        Parallel.For(0, n, opt, i =>
        {
            if (w[i] == 0f) return;
            Vector3 p = xStar[i];
            if (p.y > y0 + contactEps) return;

            float normalPush;
            if (p.y < y0) { normalPush = y0 - p.y; p.y = y0; }
            else normalPush = contactEps;

            if (frictionEnabled)
            {
                Vector3 dx = p - x[i];
                float dxN = Vector3.Dot(dx, N);
                Vector3 dxT = dx - dxN * N;
                float tLen = dxT.magnitude;
                if (tLen > 1e-12f)
                {
                    float staticLimit = muS * normalPush;
                    if (tLen <= staticLimit) p -= dxT;
                    else
                    {
                        float dynReduce = muD * normalPush;
                        float k = Mathf.Min(1f, dynReduce / tLen);
                        p -= dxT * k;
                    }
                }
            }

            p.y = y0 + slop;
            xStar[i] = p;
        });
    }

    public static float TetVolume(in Vector3 p0, in Vector3 p1, in Vector3 p2, in Vector3 p3)
    {
        Vector3 a = p1 - p0;
        Vector3 b = p2 - p0;
        Vector3 c = p3 - p0;
        return Vector3.Dot(Vector3.Cross(a, b), c) / 6f;
    }
}
