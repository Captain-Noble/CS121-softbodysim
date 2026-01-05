Shader "SoftBody/GPULitTextured"
{
    Properties
    {
        _MainTex ("Albedo", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1,1,1,1)
        _SpecColor ("Spec Color", Color) = (1,1,1,1)
        _Gloss ("Gloss (0..1)", Range(0,1)) = 0.35
        _AmbientColor ("Ambient", Color) = (0.2,0.2,0.2,1)
        _FlipNormal ("Flip Normal (0/1)", Float) = 0
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

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float3> _Normals;

            sampler2D _MainTex;
            float4 _MainTex_ST;

            float4 _BaseColor;
            float4 _SpecColor;
            float _Gloss;
            float4 _AmbientColor;
            float _FlipNormal;

            float3 _LightDirWS;
            float3 _LightColor;

            float3 _WorldSpaceCameraPos;

            float4x4 unity_ObjectToWorld;
            float4x4 unity_WorldToObject;
            float4x4 unity_MatrixVP;

            struct Attributes
            {
                uint vertexID : SV_VertexID;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 posWS : TEXCOORD0;
                float3 nrmWS : TEXCOORD1;
                float2 uv : TEXCOORD2;
            };

            Varyings vert(Attributes v)
            {
                Varyings o;

                float3 pL = _Positions[v.vertexID];
                float3 nL = _Normals[v.vertexID];
                if (_FlipNormal > 0.5) nL = -nL;

                float4 pW4 = mul(unity_ObjectToWorld, float4(pL, 1));
                o.posWS = pW4.xyz;

                float3 nW = mul((float3x3)unity_WorldToObject, nL);
                o.nrmWS = normalize(nW);

                o.uv = v.uv * _MainTex_ST.xy + _MainTex_ST.zw;

                o.positionCS = mul(unity_MatrixVP, pW4);
                return o;
            }

            float4 frag(Varyings i) : SV_Target
            {
                float3 texCol = tex2D(_MainTex, i.uv).rgb;
                float3 baseCol = texCol * _BaseColor.rgb;

                float3 N = normalize(i.nrmWS);
                float3 L = normalize(_LightDirWS);
                float3 V = normalize(_WorldSpaceCameraPos - i.posWS);
                float3 H = normalize(L + V);

                float ndl = saturate(dot(N, L));
                float shininess = lerp(8.0, 128.0, saturate(_Gloss));
                float spec = pow(saturate(dot(N, H)), shininess);

                float3 ambient = _AmbientColor.rgb * baseCol;
                float3 diffuse = ndl * _LightColor * baseCol;
                float3 specular = spec * _LightColor * _SpecColor.rgb;

                return float4(ambient + diffuse + specular, _BaseColor.a);
            }
            ENDHLSL
        }
    }
}
