import heapq

# 定义目标状态
goal_state = [1, 3, 5, 7, 0, 2, 6, 8, 4]

# 定义可行的移动方向
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 计算启发式函数的值（曼哈顿距离）
def heuristic(state):
    h = 0
    for i in range(9):
        if state[i] != 0:
            h += abs(i // 3 - (state[i] - 1) // 3) + abs(i % 3 - (state[i] - 1) % 3)
    return h

# 判断状态是否合法
def is_valid_state(state):
    inv_count = 0
    for i in range(9):
        if state[i] == 0:
            continue
        for j in range(i + 1, 9):
            if state[j] == 0:
                continue
            if state[i] > state[j]:
                inv_count += 1
    return inv_count % 2 == 0

# 找到0的位置
def find_zero(state):
    for i in range(9):
        if state[i] == 0:
            return i

# A*算法求解
def solve_puzzle(initial_state):
    if not is_valid_state(initial_state):
        return -1  # 不可解

    visited = set()
    pq = [(heuristic(initial_state), 0, initial_state)]
    while pq:
        _, steps, current_state = heapq.heappop(pq)
        if current_state == goal_state:
            return steps
        zero_pos = find_zero(current_state)
        x, y = zero_pos // 3, zero_pos % 3
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_zero_pos = new_x * 3 + new_y
                new_state = current_state[:]
                new_state[zero_pos], new_state[new_zero_pos] = new_state[new_zero_pos], new_state[zero_pos]
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    heapq.heappush(pq, (heuristic(new_state) + steps + 1, steps + 1, new_state))

lists = ['135720684', '105732684', '015732684', '135782604', '715032684']
for l in lists:
    initial_state = list(map(int, l.strip()))
    result = solve_puzzle(initial_state)
    print(result)

# # 读取输入
# initial_state = list(map(int, input().strip()))
# result = solve_puzzle(initial_state)
# print(result)