#CAUTION: some of this code was written by chatGPT because I am lazy. I checked for the cube and the result seems good though
#The clauses ensuring that every inside (called open) face is covered do not seem correct yet. Atm the SAT solver finds solutions that do not have the right number of simplices
#Right now, the code just checks all solutions until one with the right number of simplices is found.

#On desktop PC in my office I got the following times:
#2x2x1 cube (i.e. the one containing 18 lattice points): 11sec
#2x2x2 cube: 4min

#with these times we would need a computing cluster to check polytopes with morelattice points, but it seems feasible to me



import itertools as it
import numpy as np
import pycosat as sat
from scipy.spatial import ConvexHull
import itertools


#chatGPT, but seems good:
def all_unimodular_simplices_in(P):
    P = np.asarray(P, dtype=int)
    n = len(P)
    simplices = []

    for (i, j, k, l) in it.combinations(range(n), 4):
        p0, p1, p2, p3 = P[i], P[j], P[k], P[l]
        M = np.vstack((p1 - p0, p2 - p0, p3 - p0))
        det = int(round(np.linalg.det(M)))
        # Check for unimodularity: |det| == 1
        if abs(det) == 1:
            simplices.append((tuple(p0), tuple(p1), tuple(p2), tuple(p3)))

    return simplices

def segment_triangle_intersect(edge, tri, tol=1e-8):
    a,b,c = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
    x, y = np.array(edge[0]), np.array(edge[1])
    edge1 = b-a
    edge2 = c-a
    segment = x-y
    B = x-a
    M = np.column_stack([edge1, edge2, segment])
    try:
        c1,c2,c3 = np.linalg.solve(M, B)
        #print(c1,c2,c3)
    except np.linalg.LinAlgError as e:
        return False
    return tol<c1 and c1<1-tol and tol<c2 and c2<1-tol and tol<c3 and c3<1-tol and c1+c2 < 1-tol

#chatGPT, might be wrong:
def open_faces(s, P, tol=1e-8):
    P_arr = np.asarray(P, dtype=float)
    hull = ConvexHull(P_arr)
    planes = hull.equations
    open_list = []
    for face in itertools.combinations(s, 3):
        pts = np.array(face, dtype=float)
        on_boundary = False
        for (a, b, c, d) in planes:
            vals = pts.dot(np.array([a, b, c])) + d
            if np.all(np.abs(vals) <= tol):
                on_boundary = True
                break
        if not on_boundary:
            open_list.append(face)
    return open_list

def all_containing_edge(edge, simplices):
    return [i for i in range(len(simplices)) if set(edge).issubset(set(simplices[i]))]

def all_containing_tri(tri, simplices):
    return [i for i in range(len(simplices)) if set(tri).issubset(set(simplices[i]))]

def intersecting_pairs(simplices, P, tol=1e-8):
    for edge in it.combinations(P, 2):
        for tri in it.combinations(P, 3):
            if segment_triangle_intersect(edge, tri, tol):
                for i1, i2 in it.product(all_containing_edge(edge, simplices), all_containing_tri(tri, simplices)):
                    yield (i1, i2)
                
def main():
    a,b,c = 2,2,1 #dimensions of the cube, only for testing...
    polytopes = [[(x,y,z) for x in range(a+1) for y in range(b+1) for z in range(c+1)]]
    #read in the polytopes

    for P in polytopes:
        
        S = all_unimodular_simplices_in(P)
        print("Checking the following lattice polytope:")
        print(P)
        print(f"P contains {len(P)} lattice points\n")
        print(f"Number of unimodular simplices found: {len(S)}")
        
        cnf = [[x+1 for x in range(len(S))]] #the cnf instance to be solved. We already added the clause that there is at least one simplex

        ipS = intersecting_pairs(S, P)

        #intersection clauses:
        for i1, i2 in ipS:
            cnf.append([-(i1+1), -(i2+1)]) #not s1 or not s2

        print(f"Number of intersection clauses: {len(cnf)-1}")
        temp = len(cnf)
        
        #clauses so that all inside faces are covered: Does not seem to work yet, as it also finds solutions with too few simplices. But it makes finding one faster
        for (i,s) in enumerate(S):
            for face in open_faces(s, P):
                coverers = [] #list of indices of simplices that can cover the face
                for (j,s2) in enumerate(S):
                    if s2 != s and set(face).issubset(set(s2)):
                        coverers.append(j+1)
                cnf.append([-(i+1)]+coverers) #not s or the face is covered

        print(f"Number of face-covering clauses: {len(cnf)-temp}")
        print(f"Total number of clauses: {len(cnf)}\n")
        print("Start solving now...\n")

        for i, sol in enumerate(sat.itersolve(cnf)):
            if i%100 == 0: print(f"{i}: {len([x for x in sol if x>0])}-{6*a*b*c}        ", end="\r")
            if len([x for x in sol if x>0]) == 6*a*b*c: #we just check atm with the number of simplices is correct to make sure we have a triangulation. The intersection clauses should be good
                print("found a triangulation using the following simplices:")
                for s in [S[i-1] for i in sol if i>0]:
                    print(s)
                break

if __name__ == "__main__": main()

























