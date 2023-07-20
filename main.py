from dask.distributed import Client
from random import randint
from sklearn.cluster import KMeans
from math import sqrt, ceil
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


# -----------------------------
def compute_euclidean_distance_matrix(locations):
    import math

    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (int(
                    math.hypot(
                        (from_node[0] - to_node[0]),
                        (from_node[1] - to_node[1])
                    )
                ))
    return distances


def get_solution(manager, routing, solution):
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes[0]


def ortools_tsp_tour_calculator(cluster):
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2

    manager = pywrapcp.RoutingIndexManager(
        len(cluster[1]), 1, 0
    )
    routing = pywrapcp.RoutingModel(manager)
    distance_matrix = compute_euclidean_distance_matrix(cluster[1])

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        sol = get_solution(manager, routing, solution)
        sol.pop(-1)
        sol = list(map(lambda x: cluster[1][x], sol))
        return (cluster[0], sol)
    else:
        raise


def calculate_tsp_tour(cluster):
    return ortools_tsp_tour_calculator(cluster)
# -----------------------------


cities = []
n = int(input("How many cities? "))
while len(cities) != n:
    if (c := (randint(0, 2*n), randint(0, 2*n))) not in cities:
        cities.append(c)


k = ceil(sqrt(n / 2))
k = 2 if k < 2 else k
# k: number of clusters

kmeans = KMeans(n_clusters=k, random_state=0).fit(cities)

clusters, clusters_centroids = {}, {}
for i in range(k):
    clusters[i] = []
for i, c in enumerate(kmeans.cluster_centers_):
    clusters_centroids[tuple(c)] = i
for item, cluster_num in enumerate(kmeans.labels_):
    clusters[cluster_num].append(tuple(cities[item]))
# -----------------------------


client = Client('127.0.0.1:8786')
solved_clusters = client.map(calculate_tsp_tour, clusters.items())
clusters = dict(client.gather(solved_clusters))
# -----------------------------


not_merged_clusters_centroids = list(clusters_centroids.keys())
dist_matrix = euclidean_distances(not_merged_clusters_centroids)

minima, index = None, None
for i in range(len(not_merged_clusters_centroids)):
    for j in range(len(not_merged_clusters_centroids)):
        if i != j:
            if minima is None or dist_matrix[i][j] < minima:
                minima, index = dist_matrix[i][j], (i, j)

cent1, cent2 = (
    not_merged_clusters_centroids[index[0]],
    not_merged_clusters_centroids[index[1]]
)

not_merged_clusters_centroids.remove(cent1)
not_merged_clusters_centroids.remove(cent2)


def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def nearest_point_of_list_to_given_point(lst, pnt):
    result, min_dist = None, None
    for p in lst:
        dist = distance(pnt, p)
        if min_dist is None or dist < min_dist:
            min_dist, result = dist, p
    return result


cluster1_nearest_point_to_cent2, cluster2_nearest_point_to_cent1 = (
    nearest_point_of_list_to_given_point(clusters[index[0]], cent2),
    nearest_point_of_list_to_given_point(clusters[index[1]], cent1)
)

tsp_tour_clusters_order = [index[0], index[1]]


def tour_rerordering(pivot, tour: list):
    pi = tour.index(pivot)

    if distance(pivot, tour[(pi - 1) % len(tour)]) > distance(pivot, tour[(pi + 1) % len(tour)]):
        return tour[pi:] + tour[:pi]
    else:
        tour = tour[pi + 1:] + tour[:pi + 1]
        tour.reverse()
        return tour


tr = tour_rerordering(
    cluster1_nearest_point_to_cent2,
    list(clusters[index[0]])
)
tr.reverse()
clusters[index[0]] = tr

clusters[index[1]] = tour_rerordering(
    cluster2_nearest_point_to_cent1,
    list(clusters[index[1]])
)


while len(not_merged_clusters_centroids) != 0:
    rp = list(clusters[tsp_tour_clusters_order[-1]])[-1]
    lp = list(clusters[tsp_tour_clusters_order[0]])[0]

    if distance(a := nearest_point_of_list_to_given_point(not_merged_clusters_centroids, rp), rp) \
            > distance(b := nearest_point_of_list_to_given_point(not_merged_clusters_centroids, lp), lp):
        not_merged_clusters_centroids.remove(b)
        c = clusters_centroids[b]
        d = nearest_point_of_list_to_given_point(clusters[c], lp)
        tsp_tour_clusters_order = [c] + tsp_tour_clusters_order

        tr = tour_rerordering(d, list(clusters[c]))
        tr.reverse()
        clusters[c] = tr
    else:
        not_merged_clusters_centroids.remove(a)
        c = clusters_centroids[a]
        d = nearest_point_of_list_to_given_point(clusters[c], rp)
        tsp_tour_clusters_order = tsp_tour_clusters_order + [c]

        clusters[c] = tour_rerordering(d, list(clusters[c]))
# -----------------------------


def plotTSP(x, y):
    plt.plot(x, y)
    a_scale = max(x) / 100

    plt.arrow(
        x[-1], y[-1],
        (x[0] - x[-1]), (y[0] - y[-1]),
        head_width=a_scale,
        color='r', length_includes_head=True
    )

    for i in range(0, len(x)-1):
        plt.arrow(
            x[i], y[i],
            (x[i+1] - x[i]), (y[i+1] - y[i]),
            head_width=a_scale,
            color='r', length_includes_head=True
        )

    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.show()


x, y = [], []
for i in tsp_tour_clusters_order:
    for m, n in clusters[i]:
        x.append(m)
        y.append(n)

plotTSP(x, y)
