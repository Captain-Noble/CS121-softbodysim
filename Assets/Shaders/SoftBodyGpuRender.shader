Shader "SoftBody/GPUUnlit"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            // Unity built-in vars
            float4 _Color;

            StructuredBuffer<float3> _Positions;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            // Unity matrices (works in built-in + SRP)
            float4x4 unity_ObjectToWorld;
            float4x4 unity_MatrixVP;

            Varyings vert(Attributes v)
            {
                Varyings o;
                float3 pL = _Positions[v.vertexID];
                float4 pW = mul(unity_ObjectToWorld, float4(pL, 1));
                o.positionCS = mul(unity_MatrixVP, pW);
                return o;
            }

            float4 frag(Varyings i) : SV_Target
            {
                return _Color;
            }
            ENDHLSL
        }
    }
}
