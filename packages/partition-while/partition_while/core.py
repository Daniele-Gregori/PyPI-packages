def partition_while_shortest(lst, fun):
    length = len(lst)
    partitions = []

    i = 0
    while i < length:
        first = i
        last = 0

        for j in range(i, length):
            elq = fun(lst[i:j+1])
            if elq:
                last = j
            else:
                if last != 0:
                    break

        if last != 0:
            i = last + 1
            partitions.append(lst[first:last+1])
        else:
            i += 1

    return partitions


def partition_while_longest(lst, fun):
    length = len(lst)
    partitions = []

    i = 0
    while i < length:
        first = i
        last = 0

        for j in range(i, length):
            elq = fun(lst[i:j+1])
            if elq:
                last = j

        if last != 0:
            i = last + 1
            partitions.append(lst[first:last+1])
        else:
            i += 1

    return partitions


def partition_while_next(lst, fun, k):
    length = len(lst)
    partitions = []

    i = 0
    while i < length:
        first = i
        last = 0
        next_count = 0

        for j in range(i, length):
            elq = fun(lst[i:j+1])
            if elq:
                last = j
                next_count += 1
                if next_count == k:
                    break

        if last != 0:
            i = last + 1
            partitions.append(lst[first:last+1])
        else:
            i += 1

    return partitions


def PartitionWhile(lst, fun, shortest=True):
    if shortest is True:
        return partition_while_shortest(lst, fun)
    elif shortest is False:
        return partition_while_longest(lst, fun)
    elif isinstance(shortest, int):
        return partition_while_next(lst, fun, shortest)
    else:
        raise ValueError("Invalid option for shortest parameter")