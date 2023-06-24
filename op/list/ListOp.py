


class ListOp:

    def __init__(self):
        pass

    @staticmethod
    def lpad(list, n, fillvalue):
        return list + [fillvalue] * (n - len(list))
