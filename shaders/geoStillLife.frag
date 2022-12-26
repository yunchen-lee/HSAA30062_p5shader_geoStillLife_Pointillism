// Author: Yun-Chen Lee yclee@arch.nycu.edu.tw
// Project: Still A-life, Pointillism shader
// Date: 2022/12/22

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;


//=== noise ===//
//-----------------------------------------------------------------
vec3 normalMap(vec3 p, vec3 n);

vec2 random2( vec2 p ) {
    return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

//=== distance functions ===//
//-----------------------------------------------------------------
// Rotate function
mat2 Rot(float a)
{
	float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}  

float sdPlane( vec3 p, vec3 n, float h )
{
    n = normalize(n);
    return dot(p,n) + h;
}

float sdSphere( vec3 p, float s, vec3 rotation, vec3 scale )
{
    vec3 bp = p;			//	translate
    bp.yz *= Rot(rotation.x);	//	rotate
    bp.xz *= Rot(rotation.y);	//	rotate
    bp.xy *= Rot(rotation.z);	//	rotate
    bp *= scale;
    return length(bp)-s;
}

float sdBox( vec3 p, vec3 b , vec3 rotation, vec3 scale)
{
    vec3 bp = p;			//	translate
    bp.yz *= Rot(rotation.x);	//	rotate
    bp.xz *= Rot(rotation.y);	//	rotate
    bp.xy *= Rot(rotation.z);	//	rotate
    bp *= scale;
    vec3 d = abs(bp) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sdCone( vec3 p, vec2 c, float h , vec3 rotation, vec3 scale)
{
    vec3 bp = p;			//	translate
    bp.yz *= Rot(rotation.x);	//	rotate
    bp.xz *= Rot(rotation.y);	//	rotate
    bp.xy *= Rot(rotation.z);	//	rotate
    bp *= scale;
    float q = length(bp.xz);
    return max(dot(c.xy,vec2(q,bp.y)),-h-bp.y);
}

float sdCappedCylinder( vec3 p, float h, float r , vec3 rotation, vec3 scale)
{
    vec3 bp = p;			//	translate
    bp.yz *= Rot(rotation.x);	//	rotate
    bp.xz *= Rot(rotation.y);	//	rotate
    bp.xy *= Rot(rotation.z);	//	rotate
    bp *= scale;
    vec2 d = abs(vec2(length(bp.xz),bp.y)) - vec2(r,h);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdHexPrism( vec3 p, vec2 h , vec3 rotation, vec3 scale)
{
    vec3 bp = p;			//	translate
    bp.yz *= Rot(rotation.x);	//	rotate
    bp.xz *= Rot(rotation.y);	//	rotate
    bp.xy *= Rot(rotation.z);	//	rotate
    bp *= scale;
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    bp = abs(bp);
    bp.xy -= 2.0*min(dot(k.xy, bp.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(bp.xy-vec2(clamp(bp.x,-k.z*h.x,k.z*h.x), h.x))*sign(bp.y-h.x),
       bp.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float map(in vec3 p)
{
    float dist;

    float sd_plane = sdPlane(p,vec3(0.0,1.0,1.0),0.38);
    float sd_box = sdBox(p+vec3(0.0,0.2,0.0), vec3(0.4, 0.4, 0.4),vec3(0.628,0.0,0.2), vec3(2.0));
    dist = min(sd_plane,sd_box);
    
    float sd_sphere = sdSphere(p+vec3(-0.03,-0.04,-0.35), 0.2, vec3(0.0), vec3(1.0));
    dist = min(dist, sd_sphere);

    float sd_cone = sdCone(p+vec3(0.48,0.1,-0.45), vec2(0.8,0.25),0.8,vec3(-0.628,0.0,0.0), vec3(1.0));
    dist = min(dist, sd_cone);

    float sd_cappedCylinder = sdCappedCylinder(p+vec3(0.48,0.26,-0.3),0.3,0.1,vec3(0.628,0.0,-0.9), vec3(1.0));
    dist = min(dist,sd_cappedCylinder);

    float sd_hex = sdHexPrism(p+vec3(-0.45,0.45,-0.15),vec2(0.12,0.5), vec3(-0.9,-0.6,0.0), vec3(1.0));
    dist = min(dist,sd_hex);


    return dist;
}

//=== gradient functions ===//
//-----------------------------------------------------------------
vec3 gradient( in vec3 p ) //尚未normalize
{
	const float d = 0.001;
	vec3 grad = vec3(map(p+vec3(d,0,0))-map(p-vec3(d,0,0)),
                     map(p+vec3(0,d,0))-map(p-vec3(0,d,0)),
                     map(p+vec3(0,0,d))-map(p-vec3(0,0,d)));
	return grad;
}


// === raytrace functions===//
//-----------------------------------------------------------------
float trace(vec3 o, vec3 r, out vec3 p)
{
    float d=0.0, t=0.0;
    for (int i=0; i<30; ++i)
    {
        p= o+r*t;
        d=map(p);
        if(d<0.0) break;
        t += d*0.6; //影響輪廓精準程度
        }
    return t;
}

float traceInner(vec3 o,vec3 r,out vec3 p)
{
    float d=0.,t=0.01;
    for(int i=0;i<32;++i)
    {
        p=o+r*t;
        d=-map(p);
        if(d<0.001 || t>10.0) break;
        t+=d*.5;//影響輪廓精準程度
    }
    return t;
}



//=== camera functions ===//
//-----------------------------------------------------------------
mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
	vec3 cw = normalize(ta-ro);
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

// math
mat3 fromEuler(vec3 ang) {
    vec2 a1 = vec2(sin(ang.x),cos(ang.x));
    vec2 a2 = vec2(sin(ang.y),cos(ang.y));
    vec2 a3 = vec2(sin(ang.z),cos(ang.z));
    vec3 m0 = vec3(a1.y*a3.y+a1.x*a2.x*a3.x,a1.y*a2.x*a3.x+a3.y*a1.x,-a2.y*a3.x);
    vec3 m1 = vec3(-a2.y*a1.x,a1.y*a2.y,a2.x);
    vec3 m2 = vec3(a3.y*a1.x*a2.x+a1.y*a3.x,a1.x*a3.x-a1.y*a3.y*a2.x,a2.y*a3.y);
    return mat3(m0, m1, m2);
}



//=== phong shading ===//
//-----------------------------------------------------------------
// thank 仁愷!
struct Light{
    vec3 p; // position
    vec3 ia;
    vec3 id; // diffuse color
    vec3 is; // specular color
};

struct Mat{    
    float ka; // factor of abient
    float kd; // factor of diffuse
    float ks; // factor of specular
    float s; // shiness
};

vec3 phongShading(Light light, Mat m, vec3 p, vec3 n, vec3 v, out vec3 ambient, out vec3 diffuse, out vec3 specular, out float shadow){

    vec3 L = normalize(light.p-p);
    vec3 r = normalize(reflect(-L,n));

    ambient = m.ka*light.ia;
    diffuse = m.kd*dot(L,n)*light.id;
    specular = m.ks*pow(max(dot(r,v),0.0),m.s)*light.is;

    vec3 diffWithShadow = diffuse;
    vec3 temp;
    float d = trace(p+n*0.03, L, temp);
    if(d<length(light.p-p)) {
        shadow = 0.0;
        diffWithShadow *= 0.1;
    }
    else shadow = 1.0;

    vec3 result;
    // result = ambient + diffuse + specular;
    result = ambient + diffWithShadow + specular;

    return result;
    
}

//=== dot brush ===//
//-----------------------------------------------------------------
float getMask(float r, float scl, float div, vec2 uv, vec2 offset){
    // float mask = step(r*scl, length(mod((uv + offset)*div,1.)*2.-1.));
    float mask = step(r*scl,length(random2(floor((uv+offset)*div))-fract(uv*div)));
    if(mask>0.1) mask = 1.;
    else mask = 0.;
    return mask;
}

vec3 setColor(float mask, vec3 clr){
    vec3 result = vec3(1.);
    if(mask==0.) result = clr;
    return result;
}

vec3 mixColor(vec3 clr1, vec3 clr2, float ratio){
    vec3 result = vec3(1.0);

    if(clr1.r<1.|| clr1.g<1.|| clr1.b<1.){
        if(clr2 == vec3(1.)) result = clr1;
        else result = mix(clr1,clr2,ratio);
    }
    else if(clr2.r<1.|| clr2.g<1.|| clr2.b<1.){
        if(clr1 == vec3(1.)) result = clr2;
        // else result = mix(clr1,clr2, ratio);
        // else result = vec3(0.);
        // if(clr1!=vec3(1.)) result = vec3(0.);
    }
    return result;
}

//=================================================================
void main()
{
    vec2 uv = gl_FragCoord.xy/u_resolution.xy;
    uv = uv*2.0-1.0;
    uv.x*= u_resolution.x/u_resolution.y;
    uv.y*=1.0;//校正 預設值uv v軸朝下，轉成v軸朝上相同於y軸朝上為正
    vec2 mouse=(u_mouse.xy/u_resolution.xy)*2.0-1.0;

    // camera option1  (模型應在原點，適用於物件)
	// vec3 CameraRot=vec3(0.0, mouse.y, -mouse.x); 
   
    vec3 CameraRot=vec3(0.0, -0.7, 0.0);  
	vec3 ro= vec3(0.0, 0.0, 2.0)*fromEuler(CameraRot);//CameraPos;

    // vec3 CameraRot=vec3(u_time ,0.0, 0.0);  
    // vec3 CameraRot=vec3(0.0,-mouse.y*3.0, mouse.x*3.0);
	// vec3 ro= vec3(0.0, 0.0, 2.0)*fromEuler(CameraRot);//CameraPos;

	vec3 ta =vec3(0.0, 0.0, 0.0); //TargetPos; //vec3 ta =float3(CameraDir.x, CameraDir.z, CameraDir.y);//UE座標Z軸在上
	mat3 ca = setCamera( ro, ta, 0.0 );
	vec3 RayDir = ca*normalize(vec3(uv, 2.0));//z值越大，zoom in! 可替換成iMouse.z
	vec3 RayOri = ro;
	
	vec3 p,n;
	float t = trace(RayOri, RayDir, p);
	n=normalize(gradient(p));
    
    vec3 bump = normalMap(p*10.0,n);
    // // n=n+bump*0.05;
    
    
    float edge = dot(-RayDir,n);
    edge = smoothstep(-0.3,0.3,edge);
    

    // if(t<2.5) result = result; else result = vec3(0.9);//測試n, n_bump, fresnel, BG, color, fog, F, I, SS, reflectedCol

    // mask
    // float r = 1.0-(result.r+result.g+result.b)/3.0;
    // float div = 40.0;
    // float circleMask = step(r*1.2,length(mod((uv+bump.xy*0.005)*div,1.0)*2.0-1.70));
    // float circleMask = step(r*1.2,length(mod((uv)*div,1.0)*2.0-1.0));

    // float circleMask2 = step(r*1.2,length(mod((uv+vec2(0.0,0.01)+bump.xy*0.005)*div,1.0)*2.0-1.0));

    // vec3 col1 = vec3(1.0-circleMask)*vec3(0.0,0.0,1.0);
    // vec3 col2 = vec3(1.0-circleMask2)*vec3(1.0,0.0,0.0);

    // vec3 col = mix(col1,col2,0.4);

    // phong shading
    Light l1 = Light(vec3(20.*sin(u_time/3.),10.+5.*sin(u_time/5.),20.+10.*cos(u_time/3.)),vec3(.7),vec3(1.0),vec3(1.));
    Mat m1 = Mat(1.,1.,1.,1.);
    vec3 amb1, diff1, spec1, result1;
    float shdw1;
    result1 = phongShading(l1, m1, p, n, -RayDir, amb1, diff1, spec1, shdw1);

    Light l2 = Light(vec3(0.+5.*sin(u_time/3.),10.+8.*sin(u_time/4.),0.), vec3(.7),vec3(1.),vec3(1.));
    Mat m2 = Mat(1.,1.,1.,10.);
    vec3 amb2, diff2, spec2, result2;
    float shdw2;
    result2 = phongShading(l2, m2, p, n, -RayDir, amb2, diff2, spec2, shdw2);

    // SSS
    vec3 p1, n1;
    float ior = 1.31;
    vec3 RayDir1 = refract(p-l1.p,n,1.0/ior);  // material A -> B
    // vec3 RayDir1 = refract(RayDir,n,1.0/ior);  // material A -> B
    float t1 = traceInner(p,RayDir1, p1);

    // color -----------------------------------------
    // float siz = 1.3+sin(u_time)*0.5;
    float siz = 1.;
    // float mask1 = getMask(1.0-(result1.r+result1.g+result1.b)/3.0, 1.2, 40., uv, vec2(0.,0.01));
    // vec3 layer1 = setColor(mask1, vec3(1.0,0.0,0.0));

    // background
    // float mbx = getMask(0.8+bump.x*0.5, 1.0,59.0,uv,0.002*vec2(bump.y,bump.z));
    float mbx = getMask(0.3+bump.x*0.05, 1.0,58.0*siz,uv,0.002*vec2(bump.y,bump.z));
    vec3 dbx = setColor(mbx, vec3(0.67, 0.82, 1.));

    float mby = getMask(0.3+bump.y*0.05, 1.0,40.0*siz,uv,0.002*vec2(0.01+bump.x, bump.z));
    vec3 dby = setColor(mby,vec3(1., 0.89, 0.85));

    vec3 result = mixColor(dbx,dby,0.5);
    // vec3 result = mix(dbx,dby,0.5);

    float mbz = getMask(0.7+bump.z*0.5, 1.0,47.0*siz,uv,0.002*vec2(bump.x, bump.z));
    vec3 dbz = setColor(mbz,vec3(0.71, 0.96, 0.95));

    result = mixColor(result,dbz,0.5);
    
    // // toon
    vec3 rt;
    if(t<2.5)  rt = 1.0-vec3(floor(result1*2.0)/2.0);else rt = vec3(1.0);
    // vec3 rt = 1.0-vec3(floor(result1*2.0)/2.0);
    vec3 toon1 = vec3(1.), toon2 = vec3(1.), toon3 = vec3(1.);
    if(rt.x < .01) toon1 = vec3(0.0);
    if(rt.x > .2 && rt.x < .9) toon2 = vec3(0.0);
    // if(result_toon.x > .9) toon3 = vec3(.0);

    float mt1 = getMask((1.0-(toon1.x+toon1.y+toon1.z)/3.0)*0.4+bump.y*0.0, 1.0,60.0*siz,uv,0.002*vec2(0.03+bump.x, 0.05+bump.z));
    // float mt1 = getMask((1.0-(toon1.x+toon1.y+toon1.z)/3.0)*0.4+bump.y*0.0, 1.0,60.0,uv,vec2(0.));
    vec3 dt1 = setColor(mt1, vec3(0., 0.5, 0.88));
    float mt2 = getMask((1.0-(toon2.x+toon2.y+toon2.z)/3.0)*0.4+bump.y*0.0, 1.0,55.0*siz,uv,0.002*vec2(0.03+bump.y, 0.05+bump.x));
    // float mt2 = getMask((1.0-(toon2.x+toon2.y+toon2.z)/3.0)*0.4+bump.y*0.0, 1.0,55.0,uv,vec2(0.));
    vec3 dt2 = setColor(mt2, vec3(0., 0.73, 1.));
    // float mt3 = getMask((1.0-(toon3.x+toon3.y+toon3.z)/3.0)*0.4+bump.y*0.0, 1.0,45.0,uv,0.002*vec2(0.03+bump.x, 0.05+bump.z));
    // vec3 dt3 = setColor(mt3, vec3(0.0235, 1.0, 0.9529));

    vec3 temp = mixColor(dt1,dt2,0.5);
    // temp = mixColor(temp,dt3,0.5);
    result = mixColor(result,temp,0.5);

    // diffuse
    if(t<2.5)  diff1 = diff1;else diff1 = vec3(1.0);
    // float md1 = getMask((1.0-(diff1.x+diff1.y+diff1.z)/3.0)*0.4+bump.y*0.0, 1.0,69.0,uv,0.002*vec2(0.03+bump.x, 0.05+bump.z));
    float md1 = getMask((1.0-(diff1.x+diff1.y+diff1.z)/3.0)*0.4+bump.y*0.0, 1.0,69.0*siz,uv,vec2(0.));
    // vec3 dd1 = setColor(md1, vec3(0.078, 0.34, 0.22));
    vec3 dd1 = setColor(md1, vec3(0.2196, 0.1725, 0.6471));
    result = mixColor(result,dd1,0.5);


    // light2 diff
    if(t<2.5)  diff2 = diff2;else diff2 = vec3(1.0);
    // float md2 = getMask((1.0-(diff2.x+diff2.y+diff2.z)/3.0)*0.4+bump.y*0.0, 1.0,40.0,uv,0.002*vec2(0.03+bump.x, 0.05+bump.z));
    float md2 = getMask((1.0-(diff2.x+diff2.y+diff2.z)/3.0)*0.4+bump.y*0.0, 1.0,40.0*siz,uv,vec2(0.));
    vec3 dd2 = setColor(md2, vec3(0.89, 0.49, 0.27));
    result = mixColor(result,dd2,0.2);

    

    // shadow
    float msd = getMask((1.-shdw1)*0.4, 1.0,73.0*siz,uv,vec2(0.));
    vec3 dsd = setColor(msd, vec3(0.15, 0.22, 0.3));
    result = mixColor(result,dsd,0.6);

    // sss
    if (t>2.5) t1=1.;
    t1 = step(t1,0.5);
    float msss = getMask(((t1)*0.3), 1.0,60.0*siz,uv,vec2(0.));
    vec3 dsss = setColor(msss, vec3(1.0, 0.85, 0.17));
    result = mixColor(result,dsss,0.5);

    // sepcular
    float msp = getMask((length(spec1))*0.3, 1.0,75.0*siz,uv,vec2(0.));
    result += (1.-msp);

    float msp2 = getMask((length(spec2))*0.5, 1.0,75.0*siz,uv,vec2(0.));
    result += (1.-msp2);

    gl_FragColor = vec4(vec3(result),1.0);
}

//=================================================================





//=== 2d noise functions ===//
//-----------------------------------------------------------------
vec2 hash2( vec2 x )			//亂數範圍 [-1,1]
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    x = x*k + k.yx;
    return -1.0 + 2.0*fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}
float gnoise( in vec2 p )		//亂數範圍 [-1,1]
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     	    dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                	     mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     	    dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

//=== 3d noise functions p/n ===//
//-----------------------------------------------------------------
vec3 smoothSampling2(vec2 uv)
{
    const float T_RES = 32.0;
    return vec3(gnoise(uv*T_RES)); //讀取亂數函式
}

float triplanarSampling(vec3 p, vec3 n)
{
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    return  (abs(n.x)*smoothSampling2(p.yz).x
            +abs(n.y)*smoothSampling2(p.xz).x
            +abs(n.z)*smoothSampling2(p.xy).x)/fTotal;
}

const mat2 m2 = mat2(0.90,0.44,-0.44,0.90);
float triplanarNoise(vec3 p, vec3 n)
{
    const float BUMP_MAP_UV_SCALE = 0.2;
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    float f1 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.xy = m2*p.xy;
    p.xz = m2*p.xz;
    p *= 2.1;
    float f2 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    p.yx = m2*p.yx;
    p.yz = m2*p.yz;
    p *= 2.3;
    float f3 = triplanarSampling(p*BUMP_MAP_UV_SCALE,n);
    return f1+0.5*f2+0.25*f3;
}

vec3 normalMap(vec3 p, vec3 n)
{
    float d = 0.005;
    float po = triplanarNoise(p,n);
    float px = triplanarNoise(p+vec3(d,0,0),n);
    float py = triplanarNoise(p+vec3(0,d,0),n);
    float pz = triplanarNoise(p+vec3(0,0,d),n);
    return normalize(vec3((px-po)/d,
                          (py-po)/d,
                          (pz-po)/d));
}


