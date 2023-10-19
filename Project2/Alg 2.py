import heapq

def find_K_shortest_paths(N, M, K, connections):
    graph = {}
    for X, Y, D in connections:
        if X not in graph:
            graph[X] = []
        graph[X].append((Y, D))

    results = []  # 存储结果路径长度

    # 定义A*搜索函数
    def a_star():
        open_queue = []  # 优先级队列，存储节点 (路径长度, 节点, 路径, 实际距离 g(n))
        heapq.heappush(open_queue, (0, 1, [1], 0))  # 初始节点为1，路径长度为0，实际距离为0

        while open_queue:
            dist, current, path, g_value = heapq.heappop(open_queue)
            if current == N:
                results.append(g_value)  # 使用实际距离 g(n)
                if len(results) == K:
                    return results

            if current in graph:
                for neighbor, cost in graph[current]:
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        # 新的实际距离 g(n)
                        new_g_value = g_value + cost
                        heapq.heappush(open_queue, (new_g_value, neighbor, new_path, new_g_value))

    a_star()  # 执行A*搜索

    results.sort()  # 将找到的最短路径按长度排序
    for i in range(K):
        if i < len(results):
            print(results[i])
        else:
            print(-1)

# 输入数据
# 第一组输入
N1 = 5
M1 = 6
K1 = 4
connections1 = [
    (1, 2, 1),
    (1, 3, 1),
    (2, 4, 2),
    (2, 5, 2),
    (3, 4, 2),
    (3, 5, 2)
]

# 第二组输入
N2 = 6
M2 = 9
K2 = 4
connections2 = [
    (1, 2, 1),
    (1, 3, 3),
    (2, 4, 2),
    (2, 5, 3),
    (3, 6, 1),
    (4, 6, 3),
    (5, 6, 3),
    (1, 6, 8),
    (2, 6, 4)
]

# 第三组输入
N3 = 7
M3 = 12
K3 = 6
connections3 = [
    (1, 2, 1),
    (1, 3, 3),
    (2, 4, 2),
    (2, 5, 3),
    (3, 6, 1),
    (4, 7, 3),
    (5, 7, 1),
    (6, 7, 2),
    (1, 7, 10),
    (2, 6, 4),
    (3, 4, 2),
    (4, 5, 1)
]

# 第四组输入
N4 = 5
M4 = 8
K4 = 7
connections4 = [
    (1, 2, 1),
    (1, 3, 3),
    (2, 4, 1),
    (2, 5, 3),
    (3, 4, 2),
    (3, 5, 2),
    (1, 4, 3),
    (1, 5, 4)
]

#第五组输入
N5 = 6
M5 = 10
K5 = 8
connections5 = [
    (1, 2, 1),
    (1, 3, 2),
    (2, 4, 2),
    (2, 5, 3),
    (3, 6, 3),
    (4, 6, 3),
    (5, 6, 1),
    (1, 6, 8),
    (2, 6, 5),
    (3, 4, 1)
]

# 调用函数并输出结果
print('一、')
find_K_shortest_paths(N1, M1, K1, connections1)
print('二、')
find_K_shortest_paths(N2, M2, K2, connections2)
print('三、')
find_K_shortest_paths(N3, M3, K3, connections3)
print('四、')
find_K_shortest_paths(N4, M4, K4, connections4)
print('五、')
find_K_shortest_paths(N5, M5, K5, connections5)

# N, M, K = map(int, input().split())
# connections = [list(map(int, input().split())) for _ in range(M)]
#
# find_K_shortest_paths(N, M, K, connections)