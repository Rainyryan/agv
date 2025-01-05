import numpy as np

def get_orien(destination, cur_pos):
    dx = destination[0] - cur_pos[0] 
    dy = destination[1] - cur_pos[1]
    print(dx, dy)
    dist = np.abs(dx)+np.abs(dy)
    return (np.arctan2(dy, dx)*180/np.pi, dist)

print(get_orien([0.28, 0.2], [-0.17294634, -0.24295906]))