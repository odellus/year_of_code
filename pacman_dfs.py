def dfs(r, c, pacman_r, pacman_c, food_r, food_c, grid):
    pass



def test_dfs():
    raw_grid = """%%%%%%%%%%%%%%%%%%%%%--------------%---%%-%%-%%-%%-%%-%%-%-%%--------P-------%-%%%%%%%%%%%%%%%%%%%-%%.-----------------%%%%%%%%%%%%%%%%%%%%%"""
    pacman_r, pacman_c= 3, 9
    food_r, food_c = 5, 1
    r, c  = 7, 20
    grid = make_grid(raw_grid, r, c)
    print grid[r-1][c-1]

def neighbors(r, c, grid):
    if len(grid) < 1:
        return []
    grid_r = len(grid)
    grid_c = len(grid[0])
    neighbs = []
    x = grid[r][c]
    # grid[r+1][c]
    # grid[r-1][c]
    # grid[r][c+1]
    # grid[r][c-1]
    if r - 1  > 0:
        u = grid[r-1][c]
        if u != '%':
            neighbs.append(u)
    if r + 1 < len(grid):
        u = grid[r+1][c]
        if u != '%':
            neighbs.append(u)
    if c - 1 > 0:
        u = grid[r][c-1]
        if u != '%':
            neighbs.append(u)
    if c + 1 < len(grid[0]):
        u = grid[r][c+1]
        if u != '%':
            neighbs.append(u)

    return neighbs

def make_grid(raw_grid, r, c):
    grid = []
    for i in range(r):
        tmp = []
        for j in range(c):
            tmp.append(raw_grid[j + i*r])
        grid.append(tmp)
    return grid

if __name__ == "__main__":
    test_dfs()
