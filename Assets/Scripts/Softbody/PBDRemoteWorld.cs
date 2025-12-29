// PBDRemoteWorld.cs
using System;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;

public sealed class PBDRemoteWorld : MonoBehaviour
{
    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 7777;

    [Header("Time Stepping")]
    public bool simulateInFixedUpdate = true;
    public float fixedDt = 1f / 60f;

    [Header("Solver Params (Init)")]
    [Range(1, 8)] public int substeps = 2;
    [Range(1, 30)] public int solverIterations = 6;
    [Range(0.2f, 2.5f)] public float omega = 1.6f;

    public Vector3 gravity = new Vector3(0, -9.81f, 0);
    public bool groundEnabled = true;
    public float groundY = 0f;
    [Range(0f, 1f)] public float friction = 0.2f;

    [Header("XPBD Compliance")]
    public float edgeCompliance = 0.0005f;
    public float volumeCompliance = 0.0f;

    private const uint MAGIC = 0x31444250u;
    private const uint MSG_INIT = 1;
    private const uint MSG_STEP = 2;
    private const uint MSG_POSITIONS = 3;
    private const uint MSG_SHUTDOWN = 4;

    private PBDRemoteSoftBody _body;

    private Thread _netThread;
    private volatile bool _running;
    private bool _started;

    private float _requestedDt;
    private volatile bool _stepRequested;
    private volatile bool _haveNewPositions;

    private float[] _backPositions;
    private float[] _frontPositions;

    private byte[] _recvHeader = new byte[12];
    private byte[] _recvPayload;
    private byte[] _stepMsg = new byte[12 + 4];

    private readonly object _swapLock = new object();

    public void BindBody(PBDRemoteSoftBody body)
    {
        _body = body;
        if (isActiveAndEnabled && !_started) TryStartMainThread();
    }

    private void OnEnable()
    {
        _running = true;
        _started = false;

        if (_body == null) _body = FindAnyObjectByType<PBDRemoteSoftBody>();
        if (_body != null) TryStartMainThread();
    }

    private void OnDisable()
    {
        _running = false;
        try { _netThread?.Join(500); } catch { }
        _netThread = null;
        _started = false;
    }

    private void Update()
    {
        if (!simulateInFixedUpdate) Step(Time.deltaTime);
        PumpPositionsToMesh();
    }

    private void FixedUpdate()
    {
        if (!simulateInFixedUpdate) return;
        Step(fixedDt > 0f ? fixedDt : Time.fixedDeltaTime);
        PumpPositionsToMesh();
    }

    private void TryStartMainThread()
    {
        if (_started) return;

        if (_body == null)
        {
            Debug.LogError("PBDRemoteWorld: No PBDRemoteSoftBody found.");
            enabled = false;
            return;
        }

        int v = _body.VertexCount;
        var edgeIds = _body.EdgeIds;
        var tetIds = _body.TetIds;

        if (v <= 0 || edgeIds == null || edgeIds.Length == 0 || tetIds == null || tetIds.Length == 0)
        {
            Debug.LogError("PBDRemoteWorld: Body tetra data not ready.");
            enabled = false;
            return;
        }

        var x0 = _body.InitWorldPositionsBuffer;
        if (x0 == null || x0.Length != v * 3)
        {
            Debug.LogError("PBDRemoteWorld: Body init world positions not cached.");
            enabled = false;
            return;
        }

        _backPositions = new float[v * 3];
        _frontPositions = new float[v * 3];
        _recvPayload = new byte[v * 3 * 4];

        _netThread = new Thread(NetLoop) { IsBackground = true, Name = "PBDRemoteNet" };
        _netThread.Start();
        _started = true;
    }

    private void Step(float dt)
    {
        _requestedDt = dt;
        _stepRequested = true;
    }

    private void PumpPositionsToMesh()
    {
        if (!_haveNewPositions) return;
        lock (_swapLock)
        {
            if (!_haveNewPositions) return;
            _haveNewPositions = false;
            _body.ApplyWorldPositions(_frontPositions);
        }
    }

    private static void WriteU32(byte[] buf, ref int o, uint v)
    {
        buf[o++] = (byte)(v);
        buf[o++] = (byte)(v >> 8);
        buf[o++] = (byte)(v >> 16);
        buf[o++] = (byte)(v >> 24);
    }

    private static uint ReadU32(byte[] buf, ref int o)
    {
        uint v = (uint)(buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16) | (buf[o + 3] << 24));
        o += 4;
        return v;
    }

    private static void WriteF32(byte[] buf, ref int o, float v)
    {
        MemoryMarshal.Write(buf.AsSpan(o, 4), ref v);
        o += 4;
    }

    private static bool ReadExact(NetworkStream s, byte[] buf, int n)
    {
        int got = 0;
        while (got < n)
        {
            int r = s.Read(buf, got, n - got);
            if (r <= 0) return false;
            got += r;
        }
        return true;
    }

    private static void WriteExact(NetworkStream s, byte[] buf, int n)
    {
        s.Write(buf, 0, n);
    }

    private void NetLoop()
    {
        TcpClient client = null;
        NetworkStream stream = null;

        try
        {
            client = new TcpClient();
            client.NoDelay = true;
            client.Connect(host, port);
            stream = client.GetStream();

            SendInit(stream);

            bool inFlight = false;

            while (_running)
            {
                if (!inFlight && _stepRequested)
                {
                    float dt = _requestedDt;
                    _stepRequested = false;

                    int o = 0;
                    WriteU32(_stepMsg, ref o, MAGIC);
                    WriteU32(_stepMsg, ref o, MSG_STEP);
                    WriteU32(_stepMsg, ref o, 4);
                    WriteF32(_stepMsg, ref o, dt);

                    WriteExact(stream, _stepMsg, o);
                    inFlight = true;
                }

                if (inFlight)
                {
                    if (!ReadExact(stream, _recvHeader, 12)) break;
                    int ro = 0;
                    uint magic = ReadU32(_recvHeader, ref ro);
                    uint type = ReadU32(_recvHeader, ref ro);
                    uint size = ReadU32(_recvHeader, ref ro);

                    if (magic != MAGIC || type != MSG_POSITIONS) break;

                    int bytes = (int)size;
                    if (bytes != _recvPayload.Length) break;

                    if (!ReadExact(stream, _recvPayload, bytes)) break;

                    Buffer.BlockCopy(_recvPayload, 0, _backPositions, 0, bytes);

                    lock (_swapLock)
                    {
                        var tmp = _frontPositions;
                        _frontPositions = _backPositions;
                        _backPositions = tmp;
                        _haveNewPositions = true;
                    }

                    inFlight = false;
                }
                else
                {
                    Thread.Sleep(0);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"PBDRemoteWorld NetLoop error: {e}");
        }
        finally
        {
            try
            {
                if (stream != null)
                {
                    var shut = new byte[12];
                    int o = 0;
                    WriteU32(shut, ref o, MAGIC);
                    WriteU32(shut, ref o, MSG_SHUTDOWN);
                    WriteU32(shut, ref o, 0);
                    WriteExact(stream, shut, o);
                }
            }
            catch { }

            try { stream?.Close(); } catch { }
            try { client?.Close(); } catch { }
        }
    }

    private void SendInit(NetworkStream stream)
    {
        int V = _body.VertexCount;
        int[] edgeIds = _body.EdgeIds;
        int[] tetIds = _body.TetIds;

        int E = edgeIds.Length / 2;
        int T = tetIds.Length / 4;

        int[] pinned = _body.PinnedIndicesBuffer ?? Array.Empty<int>();
        uint pinnedCount = (uint)pinned.Length;

        float[] x0 = _body.InitWorldPositionsBuffer;
        if (x0 == null || x0.Length != V * 3)
            throw new InvalidOperationException("InitWorldPositionsBuffer not ready.");

        int payloadBytes =
            4 * 3 +
            4 * 2 +
            4 * 2 +
            4 * 2 +
            4 * 3 +
            4 * 1 +
            4 * 2 +
            4 * 1 +
            (int)pinnedCount * 4 +
            x0.Length * 4 +
            edgeIds.Length * 4 +
            tetIds.Length * 4;

        var buf = new byte[12 + payloadBytes];
        int o = 0;

        WriteU32(buf, ref o, MAGIC);
        WriteU32(buf, ref o, MSG_INIT);
        WriteU32(buf, ref o, (uint)payloadBytes);

        WriteU32(buf, ref o, (uint)V);
        WriteU32(buf, ref o, (uint)E);
        WriteU32(buf, ref o, (uint)T);

        WriteU32(buf, ref o, (uint)substeps);
        WriteU32(buf, ref o, (uint)solverIterations);
        WriteF32(buf, ref o, fixedDt);
        WriteF32(buf, ref o, omega);

        WriteF32(buf, ref o, edgeCompliance);
        WriteF32(buf, ref o, volumeCompliance);

        WriteF32(buf, ref o, gravity.x);
        WriteF32(buf, ref o, gravity.y);
        WriteF32(buf, ref o, gravity.z);

        WriteU32(buf, ref o, groundEnabled ? 1u : 0u);
        WriteF32(buf, ref o, groundY);
        WriteF32(buf, ref o, friction);

        WriteU32(buf, ref o, pinnedCount);
        for (int i = 0; i < pinned.Length; i++)
            WriteU32(buf, ref o, (uint)pinned[i]);

        Buffer.BlockCopy(x0, 0, buf, o, x0.Length * 4);
        o += x0.Length * 4;

        Buffer.BlockCopy(edgeIds, 0, buf, o, edgeIds.Length * 4);
        o += edgeIds.Length * 4;

        Buffer.BlockCopy(tetIds, 0, buf, o, tetIds.Length * 4);
        o += tetIds.Length * 4;

        WriteExact(stream, buf, o);
    }
}
