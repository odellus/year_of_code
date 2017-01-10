#!/usr/bin/python
def dfs( r, c, pacman_r, pacman_c, food_r, food_c, grid):
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

    d = 0
    visited = {}
    u = grid[pacman_r][pacman_c]
    path = []
    stack = [(u , pacman_r, pacman_c)]
    while stack != []:

        path.append(stack[-1])
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
                path.append(node)
                for x in path:
                    print x[1], x[2]
                return
            if node_key not in visited:
                visited[node_key] = 1
                stack.append(node)
    return



pacman_r, pacman_c = [ int(i) for i in raw_input().strip().split() ]
food_r, food_c = [ int(i) for i in raw_input().strip().split() ]
r,c = [ int(i) for i in raw_input().strip().split() ]

grid = []
for i in xrange(0, r):
    grid.append(raw_input().strip())


dfs(r, c, pacman_r, pacman_c, food_r, food_c, grid)
