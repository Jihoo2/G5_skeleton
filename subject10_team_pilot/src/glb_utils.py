"""GLB 애니메이션을 Forward Kinematics로 평가해 프레임별 Bone 월드 좌표(m, Y-up)를 계산합니다. (Three.js 없이 Python만으로 3D JSON과 같은 값을 얻음)"""
import json, struct, sys, numpy as np

CT = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NC = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT4":16}

def load(path):
    d=open(path,'rb').read()
    jl,_=struct.unpack('<II',d[12:20]); j=json.loads(d[20:20+jl])
    off=20+jl; bl,_=struct.unpack('<II',d[off:off+8]); b=d[off+8:off+8+bl]
    def acc(i):
        a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]
        o=bv.get('byteOffset',0)+a.get('byteOffset',0); n=NC[a['type']]
        dt=CT[a['componentType']]; cnt=a['count']*n
        stride=bv.get('byteStride')
        if stride and stride!=np.dtype(dt).itemsize*n:
            raw=np.frombuffer(b,np.uint8,a['count']*stride,o).reshape(a['count'],stride)
            arr=raw[:,:np.dtype(dt).itemsize*n].copy().view(dt)
        else:
            arr=np.frombuffer(b,dt,cnt,o)
        return arr.reshape(a['count'],n).astype(np.float64)
    return j,acc

def qmat(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def slerp(a,b,t):
    d=np.dot(a,b)
    if d<0: b=-b; d=-d
    if d>0.9995: r=a+t*(b-a); return r/np.linalg.norm(r)
    th=np.arccos(d); return (np.sin((1-t)*th)*a+np.sin(t*th)*b)/np.sin(th)

def sample(times,vals,t,path,interp):
    if t<=times[0]: return vals[0]
    if t>=times[-1]: return vals[-1]
    i=np.searchsorted(times,t)-1; t0,t1=times[i],times[i+1]; u=(t-t0)/(t1-t0)
    if interp=='STEP': return vals[i]
    if path=='rotation': return slerp(vals[i],vals[i+1],u)
    return vals[i]*(1-u)+vals[i+1]*u

def evaluate(path, fps=30):
    j,acc=load(path); nodes=j['nodes']
    parent={c:i for i,n in enumerate(nodes) for c in n.get('children',[])}
    skin=j['skins'][0]; joints=skin['joints']; names=[nodes[i]['name'] for i in joints]
    anim=j['animations'][0]; ch=[]
    dur=0
    for c in anim['channels']:
        s=anim['samplers'][c['sampler']]; t=acc(s['input'])[:,0]; v=acc(s['output'])
        dur=max(dur,t[-1]); ch.append((c['target']['node'],c['target']['path'],t,v,s.get('interpolation','LINEAR')))
    base={}
    for i,n in enumerate(nodes):
        base[i]=dict(translation=np.array(n.get('translation',[0,0,0]),float),
                     rotation=np.array(n.get('rotation',[0,0,0,1]),float),
                     scale=np.array(n.get('scale',[1,1,1]),float))
    nf=int(np.floor(dur*fps))+1; out=np.zeros((nf,len(joints),3))
    order=[]; seen=set()
    def visit(i):
        if i in seen: return
        if i in parent: visit(parent[i])
        seen.add(i); order.append(i)
    for i in range(len(nodes)): visit(i)
    for f in range(nf):
        t=f/fps; trs={i:dict(v) for i,v in base.items()}
        for node,p,tt,vv,ip in ch: trs[node][p]=sample(tt,vv,t,p,ip)
        W={}
        for i in order:
            m=np.eye(4); s=trs[i]
            m[:3,:3]=qmat(s['rotation'])*s['scale']; m[:3,3]=s['translation']
            W[i]=W[parent[i]]@m if i in parent else m
        out[f]=[W[i][:3,3] for i in joints]
    return names,out,dur

