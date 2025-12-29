// TetrahedralizerRuntimeBridge.cs
using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using UnityEngine;

public static class TetrahedralizerRuntimeBridge
{
    public struct TetData
    {
        public Vector3[] vertices;
        public int[] tetrahedrons;
    }

    private static Type _tetrahedralizerType;
    private static Type _tetrahedralizedMeshType;
    private static Type _tetrahedralMeshType;
    private static Type _settingsType;

    public static bool IsAvailable()
    {
        if (_tetrahedralizerType != null && _tetrahedralizedMeshType != null && _tetrahedralMeshType != null)
            return true;

        _tetrahedralizerType = FindTetrahedralizerType();
        _tetrahedralizedMeshType = FindScriptableObjectTypeByNames("TetrahedralizedMesh", "Tetrahedralized_Mesh");
        _tetrahedralMeshType = FindScriptableObjectTypeByNames("TetrahedralMesh", "Tetrahedral_Mesh");
        _settingsType = _tetrahedralizerType?.GetNestedType("Settings", BindingFlags.Public | BindingFlags.NonPublic);

        return _tetrahedralizerType != null && _tetrahedralizedMeshType != null && _tetrahedralMeshType != null;
    }

    public static bool TryTetrahedralizeMesh(
        Mesh src,
        bool remapVertexData,
        double degenerateTetrahedronRatio,
        out TetData tet,
        out string error)
    {
        tet = default;
        error = null;

        if (src == null)
        {
            error = "Input mesh is null.";
            return false;
        }

        if (!IsAvailable())
        {
            error =
                "Tetrahedralizer types not found. " +
                $"Tetrahedralizer={(_tetrahedralizerType != null)} " +
                $"TetrahedralizedMesh={(_tetrahedralizedMeshType != null)} " +
                $"TetrahedralMesh={(_tetrahedralMeshType != null)}";
            return false;
        }

        try
        {
            object tetrahedralizer = Activator.CreateInstance(_tetrahedralizerType);
            if (tetrahedralizer == null)
            {
                error = "Failed to create Tetrahedralizer instance.";
                return false;
            }

            if (_settingsType != null)
            {
                object settingsObj = CreateSettings(_settingsType, remapVertexData, degenerateTetrahedronRatio);
                if (settingsObj != null)
                {
                    MethodInfo setSettings = _tetrahedralizerType.GetMethod("SetSettings",
                        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                    if (setSettings != null)
                    {
                        setSettings.Invoke(tetrahedralizer, new[] { settingsObj });
                    }
                }
            }

            ScriptableObject tetrahedralizedMesh = ScriptableObject.CreateInstance(_tetrahedralizedMeshType);
            ScriptableObject tetrahedralMesh = ScriptableObject.CreateInstance(_tetrahedralMeshType);
            if (tetrahedralizedMesh == null || tetrahedralMesh == null)
            {
                error = "Failed to create ScriptableObject instances for tetrahedral meshes.";
                return false;
            }

            MethodInfo meshToTetrahedralizedMesh = _tetrahedralizerType.GetMethod("MeshToTetrahedralizedMesh",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (meshToTetrahedralizedMesh == null)
            {
                error = "Method MeshToTetrahedralizedMesh not found on Tetrahedralizer.";
                return false;
            }

            MethodInfo tetrahedralizedMeshToTetrahedralMesh = _tetrahedralizerType.GetMethod("TetrahedralizedMeshToTetrahedralMesh",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (tetrahedralizedMeshToTetrahedralMesh == null)
            {
                error = "Method TetrahedralizedMeshToTetrahedralMesh not found on Tetrahedralizer.";
                return false;
            }

            meshToTetrahedralizedMesh.Invoke(tetrahedralizer, new object[] { src, tetrahedralizedMesh });
            tetrahedralizedMeshToTetrahedralMesh.Invoke(tetrahedralizer, new object[] { tetrahedralizedMesh, tetrahedralMesh });

            if (!TryExtractVerticesAndTets(tetrahedralMesh, out tet.vertices, out tet.tetrahedrons, out error))
                return false;

            if (tet.vertices == null || tet.vertices.Length == 0)
            {
                error = "TetrahedralMesh.vertices is empty.";
                return false;
            }

            if (!remapVertexData && (tet.tetrahedrons == null || tet.tetrahedrons.Length == 0))
            {
                error =
                    "TetrahedralMesh.tetrahedrons is empty. " +
                    "Ensure remapVertexData=false (storing scheme one) to get tetrahedron indices.";
                return false;
            }

            return true;
        }
        catch (TargetInvocationException tie)
        {
            error = "Tetrahedralizer threw: " + (tie.InnerException != null ? tie.InnerException.Message : tie.Message);
            return false;
        }
        catch (Exception e)
        {
            error = e.GetType().Name + ": " + e.Message;
            return false;
        }
    }

    private static object CreateSettings(Type settingsType, bool remapVertexData, double ratio)
    {
        try
        {
            ConstructorInfo ctor = settingsType.GetConstructor(new[] { typeof(bool), typeof(double) });
            if (ctor != null) return ctor.Invoke(new object[] { remapVertexData, ratio });

            ctor = settingsType.GetConstructor(new[] { typeof(bool), typeof(float) });
            if (ctor != null) return ctor.Invoke(new object[] { remapVertexData, (float)ratio });

            object obj = Activator.CreateInstance(settingsType);
            if (obj == null) return null;

            FieldInfo fRemap = settingsType.GetField("remapVertexData", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                           ?? settingsType.GetField("RemapVertexData", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            FieldInfo fRatio = settingsType.GetField("degenerateTetrahedronRatio", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                           ?? settingsType.GetField("DegenerateTetrahedronRatio", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            if (fRemap != null) fRemap.SetValue(obj, remapVertexData);
            if (fRatio != null)
            {
                if (fRatio.FieldType == typeof(double)) fRatio.SetValue(obj, ratio);
                else if (fRatio.FieldType == typeof(float)) fRatio.SetValue(obj, (float)ratio);
            }

            return obj;
        }
        catch
        {
            return null;
        }
    }

    private static bool TryExtractVerticesAndTets(ScriptableObject tetrahedralMesh, out Vector3[] vertices, out int[] tets, out string error)
    {
        vertices = null;
        tets = null;
        error = null;

        object vObj = GetFieldOrPropertyValue(tetrahedralMesh, "vertices");
        object tObj = GetFieldOrPropertyValue(tetrahedralMesh, "tetrahedrons");

        if (vObj == null)
        {
            error = "TetrahedralMesh.vertices not found (field/property).";
            return false;
        }

        if (!(vObj is IList vList))
        {
            error = "TetrahedralMesh.vertices is not an IList.";
            return false;
        }

        vertices = new Vector3[vList.Count];
        for (int i = 0; i < vList.Count; i++)
            vertices[i] = (Vector3)vList[i];

        if (tObj is IList tList)
        {
            tets = new int[tList.Count];
            for (int i = 0; i < tList.Count; i++)
                tets[i] = (int)tList[i];
        }
        else
        {
            tets = Array.Empty<int>();
        }

        return true;
    }

    private static object GetFieldOrPropertyValue(object instance, string name)
    {
        if (instance == null) return null;
        Type t = instance.GetType();

        FieldInfo f = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (f != null) return f.GetValue(instance);

        PropertyInfo p = t.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (p != null) return p.GetValue(instance);

        return null;
    }

    private static Type FindTetrahedralizerType()
    {
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type[] types;
            try { types = asm.GetTypes(); }
            catch (ReflectionTypeLoadException e) { types = e.Types.Where(x => x != null).ToArray(); }
            catch { continue; }

            foreach (var t in types)
            {
                if (t == null) continue;
                if (t.Name != "Tetrahedralizer") continue;

                var m1 = t.GetMethod("MeshToTetrahedralizedMesh", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                var m2 = t.GetMethod("TetrahedralizedMeshToTetrahedralMesh", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (m1 != null && m2 != null) return t;
            }
        }
        return null;
    }

    private static Type FindScriptableObjectTypeByNames(params string[] names)
    {
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            Type[] types;
            try { types = asm.GetTypes(); }
            catch (ReflectionTypeLoadException e) { types = e.Types.Where(x => x != null).ToArray(); }
            catch { continue; }

            foreach (var t in types)
            {
                if (t == null) continue;
                if (!typeof(ScriptableObject).IsAssignableFrom(t)) continue;
                if (names.Contains(t.Name)) return t;
            }
        }
        return null;
    }
}
