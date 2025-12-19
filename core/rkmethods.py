import numpy as np

from core.RKTables import RK_COEFFICIENT_TABLES


class RKp:
    def __init__(self, order: int = 4):
        """
        Args:
            t0 (float): Starting time
            h (float): Time step
            dydt (function): function in shape  dy/dt = f(t, y)
            tmax (float, optional): Stopping time of integration. Defaults to None.
            order (int, optional): Order of RK method used. Currently implemented are {1,2,3,4,5}. Defaults to 4.
        """

        self.order = order

        # load RK coefficients
        (c, a_matrix, b) = RK_COEFFICIENT_TABLES[order]
        (self.c, self.a_matrix, self.b) = (np.array(c), np.array(a_matrix), np.array(b))

    # initialization of values before integration process begins
    def initialize(self, y0: np.ndarray, dydt):
        if len(y0) % 2 == 1:
            raise Exception("Wrong length of initial vector")
        self.y = y0.copy()
        self.ICS = y0.copy()
        self.y_history = [y0.copy()]
        self.dydt = dydt

    # integrate the given system
    def integrate(self, t0: float, h: float, tmax: float | None = None):
        self.t0 = t0
        self.t = t0
        self.h = h

        while True:
            if self.t + h < tmax:
                self.next_step()
            else:
                if tmax - self.t > 1e-14:
                    self.h = tmax - self.t
                    self.next_step()
                break

    # calculate ks for general Butcher table
    def _get_ks(self):
        k = np.zeros((len(self.c), *np.shape(self.y)))
        for i in range(len(self.c)):
            t = self.t + self.h * self.c[i]

            dy = np.zeros_like(self.y)
            for j in range(i):
                dy += self.a_matrix[i, j] * k[j]
            y = self.y + self.h * dy

            k[i] = self.dydt(t, y)
        return k.copy()

    # perform one step of RKp method
    def next_step(self):
        # obtain k values
        k = self._get_ks()
        # update y and t values
        self.y += self.h * sum(self.b[i] * k[i] for i in range(len(self.c)))
        self.t += self.h
        # save y value
        self.y_history.append(self.y.copy())
        return (self.t - self.t0) // self.h

    def get_history(self):
        return self.y_history

    @classmethod
    def get_all_implemented(cls) -> dict:
        return RK_COEFFICIENT_TABLES.keys()


if __name__ == "__main__":
    pass
