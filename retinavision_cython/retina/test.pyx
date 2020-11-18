class CheeseShop:


    def __init__(self):
        self.cheeses = []

    @property
    cpdef cheese(self):
        return "We don't have: %s" % self.cheeses

    @cheese.setter
    def cheese(self, value):
        self.cheeses.append(value)

    @cheese.deleter
    def cheese(self):
        del self.cheeses[:]

