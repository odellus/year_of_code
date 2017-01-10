def dfs(G, src, dst):

    visited = []
    path = []
    stack = [src]

    while stack != []:
        t = stack.pop(-1)
        path.append(t)
        for node in G[t]:
            if node == dst:
                path.append(dst)
                return path
            if node not in visited:
                stack.append(node)

    return -1
