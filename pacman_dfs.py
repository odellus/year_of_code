def dfs(r, c, pacman_r, pacman_c, food_r, food_c, grid):




def test_dfs():
    raw_grid = """%%%%%%%%%%%%%%%%%%%%%--------------%---%%-%%-%%-%%-%%-%%-%-%%--------P-------%-%%%%%%%%%%%%%%%%%%%-%%.-----------------%%%%%%%%%%%%%%%%%%%%%"""
    pacman_r, pacman_c= 3, 9
    food_r, food_c = 5, 1
    r, c  = 7, 20
    grid = make_grid(raw_grid, r, c)
    print grid[r-1][c-1]

# def neighbors(r, c, grid):
#     if len(grid) < 1:
#         return []
#     grid_r = len(grid)
#     grid_c = len(grid[0])
#     neighbs = []
#     x = grid[r][c]
#     # grid[r+1][c]
#     # grid[r-1][c]
#     # grid[r][c+1]
#     # grid[r][c-1]
#     if r - 1  > 0:
#         u = grid[r-1][c]
#         if u != '%':
#             neighbs.append(u)
#     if r + 1 < len(grid):
#         u = grid[r+1][c]
#         if u != '%':
#             neighbs.append(u)
#     if c - 1 > 0:
#         u = grid[r][c-1]
#         if u != '%':
#             neighbs.append(u)
#     if c + 1 < len(grid[0]):
#         u = grid[r][c+1]
#         if u != '%':
#             neighbs.append(u)
#
#     return neighbs

def make_grid(raw_grid, r, c):
    grid = []
    for i in range(r):
        tmp = []
        for j in range(c):
            tmp.append(raw_grid[j + i*r])
        grid.append(tmp)
    return grid

def search_neighb(r, c, grid):
    if len(grid) < 1:
        return []
    grid_r = len(grid)
    grid_c = len(grid[0])
    nodes = []
    if r - 1 > 0:
        u = grid[r-1][c]
        if u != '%':
            tup = (u, r-1, c)
            nodes.append(tup)
    if r + 1 < len(grid):
        u = grid[r+1][c]
        if u != '%':
            tup = (u, r+1, c)
            nodes.append(tup)
    if c - 1 > 0:
        u = grid[r][c-1]
        if u != '%':
            tup = (u, r, c-1)
            nodes.append(tup)
    if c + 1 < len(grid[0]):
        u = grid[r][c+1]
        if u != '%':
            tup = (u, r, c+1)
            nodes.append(tup)

    return nodes
def key_gen(x,y):
    return str(x) + '_' + str(y)


def dfs( r, c, pacman_r, pacman_c, food_r, food_c, grid):
    d = 0
    visited = {}
    stack = [(grid[pacman_r][pacman_c], pacman_r, pacman_c)]
    path = [(pacman_r, pacman_c)]
    while stack != []:
        t_val, t_r, t_c = stack.pop(-1)
        t_key = key_gen(t_r, t_c)
        visited[t_key] = 1 # Add to visited
        d += 1 # increment path
        # Search through neighbors.
        nodes = search_neighb(t_r, t_c, grid)
        for node in nodes:
            node_val, node_r, node_c = node
            node_key = key_gen(node_r,node_c)
            if node_val == '.':
                print d
                for x in path:
                    print x[1], x[2]
            if node_key not in visited:
                stack.append(node)
    return



pacman_r, pacman_c = [ int(i) for i in raw_input().strip().split() ]
food_r, food_c = [ int(i) for i in raw_input().strip().split() ]
r,c = [ int(i) for i in raw_input().strip().split() ]

grid = []
for i in xrange(0, r):
    grid.append(raw_input().strip())


dfs(r, c, pacman_r, pacman_c, food_r, food_c, grid)



if __name__ == "__main__":
    test_dfs()
