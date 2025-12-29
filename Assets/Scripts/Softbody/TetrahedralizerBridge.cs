using System;
using System.Collections;
using System.Reflection;
using UnityEngine;

public static class TetrahedralizerBridge
{
    private static bool _cached;

    private static Type _typeTetrahedralizer;
    private static Type _typeTetizedMesh;
    private static Type _typeTetMesh;

    private static MethodInfo _miSetSettings;
    private static MethodInfo _miMeshToTetized;
    private static MethodInfo _miTetizedToTetMesh;

    private static FieldInfo _fiVertices;
    private static FieldInfo _fiTetrahedrons;
    private static PropertyInfo _piVertices;
    private static PropertyInfo _piTetrahedrons;

    public static bool IsAvailable()
    {
        EnsureCached();
        return _typeTetrahedralizer != null &&
               _typeTetizedMesh != null &&
               _typeTetMesh != null &&
               _miMeshToTetized != null &&
               _miTetizedToTetMesh != null;
    }

    public static RuntimeTetMesh GenerateTetMeshFromUnityMesh(Mesh inputMesh, double degenerateRatio, out string error)
    {
        error = null;
        EnsureCached();

        if (!IsAvailable())
        {
            error = "Tetrahedralizer plugin types/methods not found. Ensure package is imported and compiled.";
            return null;
        }

        if (inputMesh == null)
        {
            error = "Input mesh is null.";
            return null;
        }

        try
        {
            // Create core objects
            object tetrahedralizer = Activator.CreateInstance(_typeTetrahedralizer);
            if (tetrahedralizer == null)
            {
                error = "Failed to create instance of Tetrahedralizer (Activator returned null).";
                return null;
            }

            // ScriptableObjects
            var tetized = ScriptableObject.CreateInstance(_typeTetizedMesh);
            var tetMesh = ScriptableObject.CreateInstance(_typeTetMesh);
            if (tetized == null || tetMesh == null)
            {
                error = "Failed to CreateInstance for TetrahedralizedMesh / TetrahedralMesh. Are they ScriptableObjects?";
                return null;
            }

            // Settings (optional but recommended)
            TryApplySettings(tetrahedralizer, degenerateRatio, out string settingsErr);
            // settingsErr is non-fatal; only log if you want:
            // if (!string.IsNullOrEmpty(settingsErr)) Debug.LogWarning(settingsErr);

            // Mesh -> Tetized
            _miMeshToTetized.Invoke(tetrahedralizer, new object[] { inputMesh, tetized });

            // Tetized -> TetMesh
            _miTetizedToTetMesh.Invoke(tetrahedralizer, new object[] { tetized, tetMesh });

            // Read vertices & tetrahedrons (field first, then property)
            object vertsObj = ReadMemberValue(tetMesh, _fiVertices, _piVertices);
            object tetsObj = ReadMemberValue(tetMesh, _fiTetrahedrons, _piTetrahedrons);

            if (vertsObj == null)
            {
                error = "tetrahedralMesh.vertices is null (field/property not found or not populated).";
                return null;
            }
            if (tetsObj == null)
            {
                error = "tetrahedralMesh.tetrahedrons is null (field/property not found or not populated). " +
                        "Make sure remapVertexData=false so scheme 1 is used (tetrahedrons list exists).";
                return null;
            }

            var vertices = ListVector3ToArray(vertsObj);
            var tetIds = ListIntToArray(tetsObj);

            if (vertices == null || vertices.Length == 0)
            {
                error = "Generated vertices array is empty.";
                return null;
            }
            if (tetIds == null || tetIds.Length == 0 || (tetIds.Length % 4) != 0)
            {
                error = $"Generated tetrahedrons array invalid (len={tetIds?.Length ?? -1}). Expect multiple of 4.";
                return null;
            }

            return new RuntimeTetMesh
            {
                vertices = vertices,
                tetIds = tetIds
            };
        }
        catch (TargetInvocationException tie)
        {
            // unwrap inner exception for real cause
            error = "Tetrahedralizer call failed (invocation): " + (tie.InnerException?.Message ?? tie.Message);
            return null;
        }
        catch (Exception ex)
        {
            error = "Tetrahedralizer call failed: " + ex.Message;
            return null;
        }
    }

    // ----------------- Reflection helpers -----------------
    private static object ReadMemberValue(object obj, FieldInfo fi, PropertyInfo pi)
    {
        if (fi != null) return fi.GetValue(obj);
        if (pi != null) return pi.GetValue(obj);
        return null;
    }

    private static void TryApplySettings(object tetrahedralizer, double degenerateRatio, out string err)
    {
        err = null;

        if (_miSetSettings == null) return; // plugin may omit settings

        try
        {
            // Most likely: nested type "Settings" inside Tetrahedralizer
            var settingsType = _typeTetrahedralizer.GetNestedType("Settings", BindingFlags.Public | BindingFlags.NonPublic);
            if (settingsType == null)
            {
                // fallback: parameter type of SetSettings
                settingsType = _miSetSettings.GetParameters()[0].ParameterType;
                if (settingsType == null)
                {
                    err = "SetSettings exists but cannot infer Settings type.";
                    return;
                }
            }

            // ctor(bool remapVertexData, double degenerateTetrahedronRatio)
            object settings = null;
            var ctor = settingsType.GetConstructor(new[] { typeof(bool), typeof(double) });
            if (ctor != null)
            {
                settings = ctor.Invoke(new object[] { false, degenerateRatio });
            }
            else
            {
                // some implementations might use (bool,float) or different numeric type
                var ctors = settingsType.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                foreach (var c in ctors)
                {
                    var ps = c.GetParameters();
                    if (ps.Length == 2 && ps[0].ParameterType == typeof(bool))
                    {
                        // try to coerce second param
                        object arg1 = Convert.ChangeType(degenerateRatio, ps[1].ParameterType);
                        settings = c.Invoke(new object[] { false, arg1 });
                        break;
                    }
                }
            }

            if (settings == null)
            {
                err = "Could not construct Settings(remap=false, ratio=...).";
                return;
            }

            _miSetSettings.Invoke(tetrahedralizer, new object[] { settings });
        }
        catch (Exception ex)
        {
            err = "Apply Settings failed (non-fatal): " + ex.Message;
        }
    }

    private static Vector3[] ListVector3ToArray(object listObj)
    {
        if (listObj == null) return null;

        if (listObj is IList ilist)
        {
            int n = ilist.Count;
            var arr = new Vector3[n];
            for (int i = 0; i < n; i++) arr[i] = (Vector3)ilist[i];
            return arr;
        }

        return null;
    }

    private static int[] ListIntToArray(object listObj)
    {
        if (listObj == null) return null;

        if (listObj is IList ilist)
        {
            int n = ilist.Count;
            var arr = new int[n];
            for (int i = 0; i < n; i++) arr[i] = Convert.ToInt32(ilist[i]);
            return arr;
        }

        return null;
    }

    private static void EnsureCached()
    {
        if (_cached) return;
        _cached = true;

        _typeTetrahedralizer = FindTypeByName("Tetrahedralizer");
        _typeTetizedMesh = FindTypeByName("TetrahedralizedMesh");
        _typeTetMesh = FindTypeByName("TetrahedralMesh");

        if (_typeTetrahedralizer == null || _typeTetizedMesh == null || _typeTetMesh == null) return;

        // Methods: match by name + arity (more robust across namespaces)
        _miSetSettings = FindMethod(_typeTetrahedralizer, "SetSettings", 1);

        _miMeshToTetized = FindMethod(_typeTetrahedralizer, "MeshToTetrahedralizedMesh", 2);
        _miTetizedToTetMesh = FindMethod(_typeTetrahedralizer, "TetrahedralizedMeshToTetrahedralMesh", 2);

        // Members for TetrahedralMesh: prefer FIELD per docs
        _fiVertices = _typeTetMesh.GetField("vertices", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        _fiTetrahedrons = _typeTetMesh.GetField("tetrahedrons", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

        // Fallback properties if someone wrapped them
        _piVertices = _typeTetMesh.GetProperty("vertices", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        _piTetrahedrons = _typeTetMesh.GetProperty("tetrahedrons", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
    }

    private static MethodInfo FindMethod(Type t, string name, int paramCount)
    {
        var ms = t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        for (int i = 0; i < ms.Length; i++)
        {
            if (ms[i].Name != name) continue;
            var ps = ms[i].GetParameters();
            if (ps != null && ps.Length == paramCount) return ms[i];
        }
        return null;
    }

    private static Type FindTypeByName(string typeName)
    {
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            try
            {
                var t = asm.GetType(typeName, false);
                if (t != null) return t;
            }
            catch { /* ignore */ }

            try
            {
                var types = asm.GetTypes();
                for (int i = 0; i < types.Length; i++)
                    if (types[i].Name == typeName) return types[i];
            }
            catch { /* ignore */ }
        }
        return null;
    }
}
