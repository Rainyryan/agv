import numpy as np

def projection_on_line(P1, P2, P3):
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    
    v = P2 - P1

    w = P3 - P1

    proj_w_on_v = np.dot(w, v) / np.dot(v, v) * v
    P_proj = P1 + proj_w_on_v
    return P_proj.tolist()

def get_lookahead_point(prev_p, next_p, cur_point, cur_orien):
    if prev_p.all() == next_p.all(): return next_p
    ldist = 32**0.5 #lookahead distance
    
    v1 = np.array(np.cos(np.deg2rad(cur_orien)), np.sin(np.deg2rad(cur_orien)))
    
    lahdp = cur_point+v1*ldist
    
    # print(prev_p, next_p, lahdp)
    return projection_on_line(prev_p, next_p, lahdp)

r = get_lookahead_point(np.array((0,0)), np.array((4,4)),np.array((0, -4)), 45)


s = {    
    "lpoint" : []
}
print(type(r), r)
s['lpoint'] = r

a = s['lpoint']
print(type(a), a, a[0])

print(tuple([1, 2, 3]))

print(-179%180)
ori = [0, 1]
vt = [1, 0]
M = np.array([ori,vt])
det = np.linalg.det(M)

print(det)