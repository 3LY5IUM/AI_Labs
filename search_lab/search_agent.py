import heapq
import math

def parse_map(ascii_map):
    grid = [list(line.strip()) for line in ascii_map.strip().split('\n') if line.strip()]
    start, goal = None, None
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
    return grid, start, goal

def manhattan_distance(curr, goal):
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])

def euclidean_distance(curr, goal):
    return math.sqrt((curr[0] - goal[0])**2 + (curr[1] - goal[1])**2)

def zero_heuristic(curr, goal):
    return 0

def aggressive_heuristic(curr, goal):
    return 2 * manhattan_distance(curr, goal)

def get_neighbors(state, grid):
    r, c = state
    moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
    neighbors = []
    for dr, dc, action in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] != '#':
            neighbors.append(((nr, nc), action))
    return neighbors

def search(grid, start, goal, algorithm='A*', heuristic_func=manhattan_distance):
    # Frontier stores: (priority, g_cost, state, path)
    frontier = []
    heapq.heappush(frontier, (0, 0, start, []))
    visited = set()
    states_expanded = 0

    while frontier:
        f_cost, g_cost, current_state, path = heapq.heappop(frontier)

        if current_state in visited:
            continue
            
        visited.add(current_state)
        states_expanded += 1

        if current_state == goal:
            return path, g_cost, states_expanded

        for neighbor_state, action in get_neighbors(current_state, grid):
            if neighbor_state not in visited:
                new_g_cost = g_cost + 1
                
                if algorithm == 'A*':
                    h_cost = heuristic_func(neighbor_state, goal)
                else: # BFS defaults to h=0, making it uniform cost (which is BFS for step cost 1)
                    h_cost = 0
                    
                f_cost = new_g_cost + h_cost
                new_path = path + [action]
                
                heapq.heappush(frontier, (f_cost, new_g_cost, neighbor_state, new_path))

    return None, 0, states_expanded

if __name__ == "__main__":
    # Test 1: Original Warehouse
    test1_map = """
    #################
    #S....#.......#.#
    #.###.#.#######.#
    #...#.#.......#.#
    ###.#.#######.#.#
    #...#.........#.#
    #.###########.#.#
    #.........#G#...#
    #################
    """
    
    # Test 2: Trivial
    test2_map = """
    #####
    #SG##
    #####
    """
    
    # Test 3: No Solution
    test3_map = """
    #######
    #S....#
    ###.###
    #...#G#
    #######
    """

    maps = {"Original": test1_map, "Trivial": test2_map, "No Solution": test3_map}
    
    for name, ascii_m in maps.items():
        print(f"--- Test: {name} ---")
        grid, start, goal = parse_map(ascii_m)
        path, length, expanded = search(grid, start, goal, algorithm='A*', heuristic_func=manhattan_distance)
        if path:
            print(f"Path Found: {path}")
            print(f"Length: {length}, States Expanded: {expanded}\n")
        else:
            print(f"No plan found. States Expanded: {expanded}\n")
            
    print("--- Algorithm Comparison (Original Map) ---")
    grid, start, goal = parse_map(test1_map)
    
    # BFS
    path, length, expanded = search(grid, start, goal, algorithm='BFS')
    print(f"BFS -> Length: {length}, Expanded: {expanded}")
    
    # A* variants
    heuristics = {
        "Manhattan": manhattan_distance,
        "Zero (Dijkstra)": zero_heuristic,
        "Euclidean": euclidean_distance,
        "Aggressive (2x Manhattan)": aggressive_heuristic
    }
    
    for h_name, h_func in heuristics.items():
        path, length, expanded = search(grid, start, goal, algorithm='A*', heuristic_func=h_func)
        print(f"A* ({h_name}) -> Length: {length}, Expanded: {expanded}")
