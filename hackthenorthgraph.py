##
## ========================================================== ##
##  Akhar, Ammad, Bryan, Dhir
##  Hackthenorth 2022
##  Graphify
## ========================================================== ##
##

'''
Note: this program is based off of University of Waterloo
      MATH239 definitions and requirements
'''

# import django
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

# ========================definition======================== #
v = set()
Nv = 0
e = set()
Ne = 0
cycle = 0
isTree = bool()
isForest = bool() 
isBipartite = bool()
isPlanar = bool()
bridges = set()
Nbridge = 0
chroma = 2
isConnected = bool()
# ========================================================== #


# =====================image processing===================== #

# read image
img = cv2.imread('g6.png')

# blurring and transitioning image to gray scale
img_gr = cv2.GaussianBlur((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), (5,5), 0)

# image of edges after running Canny
dst = cv2.Canny(img_gr, 50, 200, None, 3)

# copy edges to the images that will display the results in BGR
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

# find circles
circles = cv2.HoughCircles(img_gr, cv2.HOUGH_GRADIENT_ALT, 1, 10, param1=300, param2=0.85)

# if circle(s) found
if circles is not None: 
   # get position and radius as int
   circles = np.round(circles[0, :]).astype("int").tolist()
   
   # node index and relative position
   nodes = dict()
   for i in range(len(circles)):
      nodes[i] = circles[i]
   # save node indices as vertex set
   v = set(nodes.keys())

   # loop over the circles
   for (x, y, r) in circles:
      # highlight circumference(s)
      cv2.circle(cdstP, (x, y), r, (0, 255, 0), 2)
      # highlight center(s)
      cv2.circle(cdstP, (x, y), 1, (0, 0, 255), 3)

def match(point):
   '''
   match: listof(int int) -> int

   approximates a given point to a node by calculating
   relative distance with a tolerance of 15 pixels
   '''
   val = list(nodes.values())
   key = list(nodes.keys())
   for i in range(len(circles)):
      d = math.sqrt((point[0]-circles[i][0])**2+(point[1]-circles[i][1])**2)
      radius = circles[i][2]
      if d < (radius+15):
         return key[val.index(circles[i])]

# find lines
linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 50, None, 9, 50)

# if line(s) found
if linesP is not None:
   # loop over lines
   for i in range(len(linesP)):
      l = linesP[i][0]

      # node start
      start = match([l[0],l[1]])
      # node end
      end = match([l[2],l[3]])
      edge = tuple([start, end])
      if (start != None) and (end != None) and (start != end):
         # add connection to set if both ends exist
         e.add(edge)

      # highlight edges
      cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)

 # show the output image
cv2.imshow("Detected Lines - Probabilistic Line Transform", cdstP) 

# ========================================================== #

# create graph
G = nx.Graph()
G.add_nodes_from(list(v))
G.add_edges_from(list(e))
print('vertex set: ', v)
print('edge set: ', e)

# node amount, edge amount
Nv = len(v)
Ne = len(e)
print('|V(G)|: ', Nv)
print('|E(G)|: ', Ne)

# connected
print('\n--------connectivity---------')
isConnected = nx.is_connected(G)
if not isConnected:
   components = list(nx.connected_components(G))
   Nc = len(components)
else:
   components = list(e)
   Nc = 1
print('components: ', components)
print('number of components', Nc)

# tree
print('\n------------tree-------------')
if Nv - 1 == Ne and isConnected:
   isTree = True
   cycle = 0
   isBipartite = True
   chroma = 2
print('G is a tree: ', isTree)

if Nv - Nc == Ne and not isConnected:
   isForest = True
   cycle = 0
   isBipartite = True
   chroma = 2
print('G is a forest: ', isForest)

# bipartite
isBipartite = bipartite.is_bipartite(G)

# bridges
print('\n-----------bridge------------')
bridges = list(nx.bridges(G))
Nbridge = len(bridges)
if not bridges:
   print('list of bridges: N/A')
else:
   print('list of bridges: ', list(nx.bridges(G)))
print('number of bridges: ', Nbridge)

# degree
print('\n-----------degree------------')
deg = {}
for vertex in range(Nv):
    deg[vertex] = sum(vertex in elem for elem in e)
mindeg = min(list(deg.values()))
maxdeg = max(list(deg.values()))
print('degree(s) of each vertex: ', deg)
print('min degree: ', mindeg)
print('max degree: ', maxdeg)

# coloring/chromatic number
print('\n----------coloring-----------')
if isBipartite:
   if isConnected:
      A,B = bipartite.sets(G)
   else:
      A = set()
      B = set()
      for c in components:
         H = G.subgraph(c)
         x,y = bipartite.sets(H)
         A |= x
         B |= y
   chroma = 2
   print('G is bipartite: ', isBipartite)
   print('set A: ', A)
   print('set B: ', B)
   coloring = {'A':0,'B':1}
else:
   coloring = nx.coloring.greedy_color(G, strategy="largest_first")
   chroma = len(set(coloring.values()))
print('possible coloring: ', coloring)
print('chromatic number: ', chroma)

# girth
print('\n-----------girth-------------')
cycle_basis = list(nx.cycle_basis(G))
girth_graph = min(cycle_basis, key = len) if len(cycle_basis) > 0 else None
girth = len(girth_graph) if girth_graph != None else 'infty'
print('cycle basis: ', cycle_basis)
print('shortest cycle (one of): ', girth_graph)
print('girth length: ', girth)

def get_counterexample(graph):
   # copy graph
   graph = nx.Graph(graph)
   # find Kuratowski subgraph
   subgraph = nx.Graph()
   for u in graph:
      nbrs = list(graph[u])
      for v in nbrs:
         graph.remove_edge(u, v)
         if nx.check_planarity(graph)[0]:
            graph.add_edge(u, v)
            subgraph.add_edge(u, v)
   return subgraph

# planarity
print('\n-----------planar------------')
isPlanar, embedding = nx.check_planarity(G)
print('G is planar: ', isPlanar)
print('planar embedding: ', embedding)
if not isPlanar:
   counterexample = get_counterexample(G)
   if len(counterexample.nodes) == 5:
      print('Kuratowski: K5 complete graph')
   else:
      print('Kuratowski: K33 bipartite graph')

# plotting graph w node labels using matplot
plt.figure()
plt.title('Detected Graph Redrawn')
lab = {}
for idx in range(len(v)):
   lab[idx] = idx
nx.draw_networkx(G, labels = lab)
plt.show()

cv2.waitKey(0)