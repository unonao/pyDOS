import snap
""""""
if __name__ == '__main__':
    # generate a network using Forest Fire model
    G = snap.GenForestFire(1000, 0.35, 0.35)
    print("G: Nodes %d, Edges %d" % (G.GetNodes(), G.GetEdges()))
    snap.SaveEdgeList(UGraph, 'forest_fire_%d_%f_%f.txt' % (1000, 0.35, 0.35))
