from collections import deque

def parse_warehouse(ascii_map):
    """Parses the ASCII representation of the warehouse."""
    grid = [list(line.strip()) for line in ascii_map.strip().split('\n') if line.strip()]
    start, goal = None, None
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
    return grid, start, goal

def get_valid_moves(grid, current_pos):
    """Returns a list of valid (row, col) positions and the action taken to get there."""
    r, c = current_pos
    moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
    valid = []
    
    for dr, dc, action in moves:
        nr, nc = r + dr, c + dc
        # Check boundaries and obstacles
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] != '#':
                valid.append(((nr, nc), action))
    return valid

def goal_based_search(grid, start, goal):
    """
    Implements a Breadth-First Search (BFS) for the goal-based agent.
    BFS is chosen because it guarantees the shortest path in an unweighted grid.
    """
    if not start or not goal:
        return None

    # Queue stores tuples of (current_position, path_of_actions)
    queue = deque([(start, [])])
    visited = set([start])

    while queue:
        current_pos, path = queue.popleft()

        if current_pos == goal:
            return path

        for next_pos, action in get_valid_moves(grid, current_pos):
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [action]))

    return None

if __name__ == "__main__":
    # Note: Cleaned up the ASCII map to ensure a uniform rectangular grid 
    # based on the fragmented components in the document prompt.
    warehouse_map = """
    #####################
    #S....#.......G#....#
    #.##.....##########.#
    #....##..........#..#
    #.######.###.#.###..#
    #.......#........#..#
    #####################
    """

    print("--- Warehouse Environment ---")
    print(warehouse_map.strip())
    
    grid, start, goal = parse_warehouse(warehouse_map)
    
    if not start or not goal:
        print("Error: Start (S) or Goal (G) not found in the map.")
    else:
        print(f"\nStart Position: {start}")
        print(f"Goal Position: {goal}")
        
        print("\nInitiating Goal-Based Search (BFS)...")
        path = goal_based_search(grid, start, goal)
        
        if path:
            print(f"Success! Collision-free path found:")
            print(f"Path sequence ({len(path)} steps): {path}")
        else:
            print("Failed. No collision-free path exists.")
