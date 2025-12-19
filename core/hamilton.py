import numpy as np


################################
#  GENERAL CLASS SHAPE
################################
class Hamiltonian:
    @classmethod
    def history_of_kinetic_energies(
        cls, masses: np.ndarray, momenta: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def history_of_total_potential_energy(
        cls, masses: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def kinetic_energies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def kinetic_energy(cls, masses: np.ndarray, momenta: np.ndarray) -> float:
        raise NotImplementedError()

    @classmethod
    def total_potential_energy(cls, masses: np.ndarray, positions: np.ndarray) -> float:
        raise NotImplementedError()

    @classmethod
    def momentum_derivative(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def position_derivative(
        cls, masses: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()


################################
#  Central motion hamiltonian
################################


class KeplerHamiltonian(Hamiltonian):
    @classmethod
    def history_of_momenta(
        cls, momenta: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        return np.sum(
            (
                (momenta * momenta).sum(axis=2) / 2
                - 1 / np.linalg.norm(positions, ord=2, axis=2)
            ),
            axis=1,
        )

    @classmethod
    def history_of_values(
        cls, momenta: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        temp = momenta * positions[:, :, ::-1]
        return temp[:, :, 0] - temp[:, :, 1]

    @classmethod
    def momentum_derivative(cls, momenta: np.ndarray) -> np.ndarray:
        return momenta

    @classmethod
    def position_derivative(cls, positions: np.ndarray) -> np.ndarray:
        return positions / (np.linalg.norm(positions, ord=2, axis=1) ** 3)


################################
#  General Newtonian hamiltonian
################################
class NewtonHamiltonian(Hamiltonian):
    @classmethod
    def history_of_kinetic_energies(
        cls, masses: np.ndarray, momenta: np.ndarray
    ) -> np.ndarray:
        res = np.zeros(shape=momenta.shape[: len(momenta.shape) - 1])
        for i in range(len(momenta)):
            res[i] = cls.kinetic_energies(masses=masses, momenta=momenta[i])
        return res

    @classmethod
    def history_of_total_potential_energy(
        cls, masses: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        res = np.zeros(shape=len(positions))
        for i in range(len(positions)):
            res[i] = cls.total_potential_energy(masses=masses, positions=positions[i])
        return res

    @classmethod
    def kinetic_energies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(np.sum, axis=1, arr=(momenta**2)) / (2 * masses)

    @classmethod
    def kinetic_energy(cls, masses: np.ndarray, momenta: np.ndarray) -> float:
        return cls.kinetic_energies(masses, momenta).sum()

    @classmethod
    def total_potential_energy(cls, masses: np.ndarray, positions: np.ndarray) -> float:
        result = 0
        for j in range(1, len(positions)):
            temp = 0
            for i in range(j):
                temp -= masses[i] / np.linalg.norm(positions[i] - positions[j], ord=2)

            temp *= masses[j]
            result += temp
        return result

    @classmethod
    def momentum_derivative(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        return momenta / np.transpose(np.array([masses]))

    @classmethod
    def position_derivative(
        cls, masses: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        # We calculate all the differences between the bodies
        pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        # transform them into distances
        distances = np.linalg.norm(pos_diff, axis=2)
        distances[distances == 0] = np.inf

        # obtaining directions
        directions = pos_diff / distances[:, :, np.newaxis]

        # we calculate the resulting values of dH/dq
        contributions = (
            masses[:, np.newaxis] * directions / distances[:, :, np.newaxis] ** 2
        )
        result = masses[:, np.newaxis] * contributions.sum(axis=1)

        return result

    @classmethod
    def position_derviative_dumdum(
        cls, masses: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        result = []
        for i in range(len(positions)):
            temp = 0
            for j in range(len(positions)):
                if j == i:
                    continue

                temp -= (
                    (positions[i] - positions[j])
                    * masses[j]
                    / pow(np.linalg.norm(positions[i] - positions[j]), 3)
                )

            temp *= masses[i]
            result.append(temp)
        return np.array(result)


if __name__ == "__main__":
    a = NewtonHamiltonian
    masses = np.array([4, 4, 4, 4, 4])
    positions = np.array(
        [
            np.array([0, 0]),
            np.array([30, 4]),
            np.array([5, -60]),
            np.array([7, 80]),
            np.array([9, 100]),
        ]
    )
    momenta = np.array(
        [
            np.array([1, 2]),
            np.array([3, 10]),
            np.array([5, 90]),
            np.array([7, 8]),
            np.array([9, 10]),
        ]
    )

    print("Kinetic energies")
    print(a.kinetic_energies(masses, momenta))

    print("Total kinetic energy")
    print(a.kinetic_energy(masses, momenta))

    print("Total potential energy")
    print(a.total_potential_energy(masses, positions))

    print("dH / dp")
    print(a.momentum_derivative(masses, momenta))

    print("dH / dq")
    print(a.position_derivative(masses, positions))

    print("dH / dq dumdum way")
    print(a.position_derviative_dumdum(masses, positions))

    ###############################################################
    ##  Performance testing
    ###############################################################
    test_times = True

    if test_times:
        import time

        samples = 1000

        masses = np.random.randint(1, 1000, samples)
        positions = np.array([np.random.randint(1, 100000, 2) for i in range(samples)])
        momenta = np.array([np.random.randint(1, 100000, 2) for i in range(samples)])

        ### Test dHdp

        ti_momentum_derivative = time.time()
        a.momentum_derivative(masses, momenta)
        tf_momentum_derivative = time.time()

        ### Test dHdq

        ti_position_derivative = time.time()
        a.position_derivative(masses, positions)
        tf_position_derivative = time.time()

        ### Test position_derviative_dumdum

        ti_position_derviative_dumdum = time.time()
        # a.position_derviative_dumdum(masses,positions)
        tf_position_derviative_dumdum = time.time()

        print(
            f"Runtimes on {samples} samples in 2D: \
            \n dHdp: {tf_momentum_derivative - ti_momentum_derivative} seconds\
            \n dHdq: {tf_position_derivative - ti_position_derivative} seconds\
            \n dHdqDUMDUM: {tf_position_derviative_dumdum - ti_position_derviative_dumdum} seconds"
        )

        masses = np.random.randint(1, 1000, samples)
        positions = np.array([np.random.randint(1, 100000, 3) for i in range(samples)])
        momenta = np.array([np.random.randint(1, 100000, 3) for i in range(samples)])

        ### Test dHdp

        ti_momentum_derivative = time.time()
        a.momentum_derivative(masses, momenta)
        tf_momentum_derivative = time.time()

        ### Test dHdq
        ti_position_derivative = time.time()
        a.position_derivative(masses, positions)
        tf_position_derivative = time.time()

        ### Test dHdqDUMDUM
        ti_position_derviative_dumdum = time.time()
        # a.dHdqDUMDUM(masses,positions)
        tf_position_derviative_dumdum = time.time()

        print(
            f"Runtimes on {samples} samples in 3D: \
            \n dHdp: {tf_momentum_derivative - ti_momentum_derivative} seconds\
            \n dHdq: {tf_position_derivative - ti_position_derivative} seconds\
            \n dHdqDUMDUM: {tf_position_derviative_dumdum - ti_position_derviative_dumdum} seconds"
        )
