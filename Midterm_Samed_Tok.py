import osmnx as ox
import heapq
import numpy as np

# ────────────────────────────────────────────────────────────────
# 1) Download and clean the map
place_name = "Küçükçekmece, Istanbul, Turkey"
G = ox.graph_from_place(place_name, network_type="drive")

for edge in G.edges:
    maxspeed = 40  # default speed if missing
    if "maxspeed" in G.edges[edge]:
        maxspeed = G.edges[edge]["maxspeed"]
        if isinstance(maxspeed, list):
            speeds = [int(speed) for speed in maxspeed if speed.isdigit()]
            if speeds:
                maxspeed = min(speeds)
            else:
                maxspeed = 40
        elif isinstance(maxspeed, str):
            try:
                maxspeed = int(maxspeed)
            except:
                maxspeed = 40
    G.edges[edge]["maxspeed"] = maxspeed
    G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed

# ────────────────────────────────────────────────────────────────
# 2) Visual styling functions
def style_unvisited_edge(edge):
    G.edges[edge]["color"] = "#cccccc"   # light gray
    G.edges[edge]["alpha"] = 0.5
    G.edges[edge]["linewidth"] = 0.4

def style_visited_edge(edge):
    G.edges[edge]["color"] = "#0066cc"   # blue
    G.edges[edge]["alpha"] = 0.9
    G.edges[edge]["linewidth"] = 1.2

def style_active_edge(edge):
    G.edges[edge]["color"] = "#ff9900"   # bright orange
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1.5

def style_path_edge(edge):
    G.edges[edge]["color"] = "red"       # red path = stands out
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 2.5

def plot_graph():
    ox.plot_graph(
        G,
        node_size=[G.nodes[node]["size"] for node in G.nodes],
        edge_color=[G.edges[edge]["color"] for edge in G.edges],
        edge_alpha=[G.edges[edge]["alpha"] for edge in G.edges],
        edge_linewidth=[G.edges[edge]["linewidth"] for edge in G.edges],
        node_color="black",             # so they stand out on white
        bgcolor="white"                 # HIGH contrast background
    )

# ────────────────────────────────────────────────────────────────
# 3) Algorithm Implementations
def dijkstra(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)

    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            if plot:
                print(f"Dijkstra steps: {step}")
                plot_graph()
            return
        if G.nodes[node]["visited"]:
            continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

def distance(node1, node2): # heuristic value
    x1, y1 = G.nodes[node1]["x"], G.nodes[node1]["y"]
    x2, y2 = G.nodes[node2]["x"], G.nodes[node2]["y"]
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def a_star(orig, dest, plot=False):
    for node in G.nodes:
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
        G.nodes[node]["g_score"] = float("inf")
        G.nodes[node]["f_score"] = float("inf")
    for edge in G.edges:
        style_unvisited_edge(edge)

    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    G.nodes[orig]["g_score"] = 0
    G.nodes[orig]["f_score"] = distance(orig, dest)
    pq = [(G.nodes[orig]["f_score"], orig)]
    step = 0

    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            if plot:
                print(f"A* steps: {step}")
                plot_graph()
            return
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            tentative_g_score = G.nodes[node]["g_score"] + distance(node, neighbor)
            if tentative_g_score < G.nodes[neighbor]["g_score"]:
                G.nodes[neighbor]["previous"] = node
                G.nodes[neighbor]["g_score"] = tentative_g_score
                G.nodes[neighbor]["f_score"] = tentative_g_score + distance(neighbor, dest)
                heapq.heappush(pq, (G.nodes[neighbor]["f_score"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1

def reconstruct_path(orig, dest, plot=False):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        style_path_edge((prev, curr, 0))
        curr = prev
    if plot:
        print(f"Path Distance: {dist/1000:.2f} km")
        plot_graph()

# ────────────────────────────────────────────────────────────────
# 4) Experiment
# Instead of totally random, let's **force good start/end points** manually

# Find good start/end near major roads
start_lon, start_lat = 28.775, 41.002
end_lon, end_lat     = 28.792, 41.020

start = ox.nearest_nodes(G, start_lon, start_lat)
end   = ox.nearest_nodes(G, end_lon,   end_lat)

print(f"Start node {start}, End node {end}")

# Dijkstra
print("Running Dijkstra...")
dijkstra(start, end, plot=True)
reconstruct_path(start, end, plot=True)

# A*
print("Running A*...")
a_star(start, end, plot=True)
reconstruct_path(start, end, plot=True)

# ────────────────────────────────────────────────────────────────
# 4)  Animated comparison  (bright colours, white background)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# pick start/end once
start = ox.nearest_nodes(G, 28.775, 41.002)   # change coords if you like
end   = ox.nearest_nodes(G, 28.792, 41.020)

print(f"Start {start} ➜ End {end}")

# ---------------------------------------------------------------
# helpers to capture exploration order for each algorithm
def capture_dijkstra():
    explored = []
    for n in G.nodes: G.nodes[n]["previous"] = None
    pq = [(0, start)]
    dist = {start: 0}
    seen = set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u); explored.append(u)
        if u == end: break
        for _, v, k in G.out_edges(u, keys=True):
            w = G.edges[(u, v, k)]["weight"]
            alt = dist[u] + w
            if v not in dist or alt < dist[v]:
                dist[v] = alt; G.nodes[v]["previous"] = u
                heapq.heappush(pq, (alt, v))
    return explored

def capture_astar():
    explored = []
    for n in G.nodes: G.nodes[n]["previous"] = None
    g = {start: 0}
    f = {start: distance(start, end)}
    pq = [(f[start], start)]
    while pq:
        _, u = heapq.heappop(pq)
        explored.append(u)
        if u == end: break
        for _, v, k in G.out_edges(u, keys=True):
            alt = g[u] + distance(u, v)
            if v not in g or alt < g[v]:
                g[v] = alt
                f[v] = alt + distance(v, end)
                G.nodes[v]["previous"] = u
                heapq.heappush(pq, (f[v], v))
    return explored

explored_d  = capture_dijkstra()
explored_a  = capture_astar()

# rebuild final path for each
def rebuild_path():
    path = []
    cur  = end
    while cur != start:
        path.append((G.nodes[cur]['x'], G.nodes[cur]['y']))
        cur = G.nodes[cur]["previous"]
    path.append((G.nodes[start]['x'], G.nodes[start]['y']))
    path.reverse()
    xs, ys = zip(*path)
    return xs, ys

path_dx, path_dy = rebuild_path()            # after A* call, nodes hold A* predecessors

# ---------------------------------------------------------------
#  build animation
fig, ax = ox.plot_graph(
    G, node_size=0, edge_color="#cccccc", edge_alpha=0.6, edge_linewidth=0.5,
    bgcolor="white", show=False, close=False
)

sc_d  = ax.scatter([], [], c="#0066cc", s=10, zorder=3, label="Dijkstra explored")
sc_a  = ax.scatter([], [], c="#ff9900", s=10, zorder=4, label="A* explored")
(line,) = ax.plot([], [], c="red", lw=2.5, zorder=5, label="A* path")

total_frames = max(len(explored_d), len(explored_a))
interval_ms  = 40                           # ≈ 25 fps classroom-smooth

import numpy as np

def update(frame):
    # helper → always return an (N,2) array, even if N = 0
    def to_xy(node_list):
        if not node_list:
            return np.empty((0, 2))
        return np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in node_list])

    # current slices
    d_pts = explored_d[:min(frame, len(explored_d))]
    a_pts = explored_a[:min(frame, len(explored_a))]

    # update scatters safely
    sc_d.set_offsets(to_xy(d_pts))
    sc_a.set_offsets(to_xy(a_pts))

    # draw final A* path on or after the last A* frame
    if frame >= len(explored_a):
        line.set_data(path_dx, path_dy)

    return sc_d, sc_a, line


ani = animation.FuncAnimation(
    fig, update, frames=total_frames, interval=interval_ms,
    blit=True, repeat=False
)

# ---------------------------------------------------------------
# save movie  (MP4 preferred; GIF fallback)
out_mp4 = Path("astar_vs_dijkstra.mp4")
try:
    from matplotlib.animation import FFMpegWriter
    ani.save(out_mp4, writer=FFMpegWriter(fps=1000//interval_ms, codec="libx264"))
    print("✅  saved", out_mp4)
except (ImportError, animation.MovieWriterError):
    ani.save(out_mp4.with_suffix(".gif"), writer="pillow")
    print("⚠️  FFmpeg not found – saved GIF instead")