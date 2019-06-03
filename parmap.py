"""Defines function parmap, a parallel version of map."""
import multiprocessing as mp
import gc


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=None):
    """Parallelizes the computation of 'map(f, X)' over 'nprocs' processors,
    where f is a function, X is an iterable. Returns list of length len(X)."""
    if nprocs is None:
        nprocs = mp.cpu_count()
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    out = [x for i, x in sorted(res)]
    del(proc, res, sent, q_in, q_out)
    gc.collect()
    return out
