import numpy as np
import time
import matplotlib.pyplot as plt

def _movie_picker(flight_length, movie_lengths):
    """
    We're give a flight_length and a list of flim lengths.
    We want to find a pair of movies that equal the flight time.
    We will return the index of two movies who's combined length
    is equal to the flight length.
    """

    movie_idx = {x: k for (k, x) in enumerate(movie_lengths)}
    for val, idx in movie_idx.items():
        comp = flight_length - val
        if comp in movie_idx:
            comp_idx = movie_idx[comp]
            if comp_idx == idx: continue
            return (idx, comp_idx)



def movie_picker(flight_length, movie_lengths):
    """
    We're give a flight_length and a list of flim lengths.
    We want to find all pairs of movies that equal the flight time.
    We will return the index of two movies who's combined length
    is equal to the flight length.
    """

    movie_idx = {x: k for (k, x) in enumerate(movie_lengths)}
    res = []
    for val, idx in movie_idx.items():
        comp = flight_length - val
        if comp in movie_idx:
            comp_idx = movie_idx[comp]
            # Don't want to show same moview twice
            if comp_idx == idx: continue 
            res.append((idx, comp_idx))
    return res


flight_length = 240
movie_lengths = [120, 110, 130, 100, 98, 112]

picks = movie_picker(flight_length, movie_lengths)
print(picks)

def get_movie_lengths(size):
    return np.random.randint(80, 200, size=(size,)).tolist()

picks = movie_picker(flight_length, movie_lengths)
print(picks)

def time_algo(flight_length=240):
    sizes = [1000000*(k+1) for k in range(10)]
    times = []
    for size in sizes:
        movie_lengths = get_movie_lengths(size)
        t = time.time()
        picks = movie_picker(flight_length, movie_lengths)
        times.append(time.time()-t)
    idxs = [x for x in range(10)]
    plt.plot(idxs, times)
    plt.show()

if __name__ == '__main__':
    time_algo()