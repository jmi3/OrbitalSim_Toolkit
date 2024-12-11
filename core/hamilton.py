import numpy as np


################################
#  GENERAL CLASS SHAPE
################################
class Hamiltonian:
    @classmethod
    def HistoryOfKineticEnergies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @classmethod
    def HistoryOfTotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def KineticEnergies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def KineticEnergy(cls, masses: np.ndarray, momenta: np.ndarray) -> float:
        raise NotImplementedError()

    @classmethod
    def TotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> float:
        raise NotImplementedError()
    
    @classmethod
    def dHdp(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @classmethod
    def dHdq(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

################################
#  Central motion hamiltonian
################################

class KeplerHamiltonian(Hamiltonian):
    @classmethod
    def HistoryOfLp(cls, momenta: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return np.sum(((momenta*momenta).sum(axis=2)/2 - 1/np.linalg.norm(positions,ord=2,axis=2)),axis=1)

    @classmethod
    def HistoryOfValues(cls, momenta: np.ndarray, positions: np.ndarray) -> np.ndarray:
        temp = momenta*positions[:,:,::-1]
        return  temp[:,:,0] - temp[:,:,1]

    @classmethod
    def dHdp(cls, momenta: np.ndarray) -> np.ndarray:
        return momenta
    
    @classmethod
    def dHdq(cls, positions: np.ndarray) -> np.ndarray:
        return positions/(np.linalg.norm(positions,ord=2,axis=1)**3)




################################
#  General Newtonian hamiltonian
################################
class NewtonHamiltonian(Hamiltonian):
    @classmethod
    def HistoryOfKineticEnergies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=momenta.shape[:len(momenta.shape)-1])
        for i in range(len(momenta)):
            res[i] = cls.KineticEnergies(masses=masses,momenta=momenta[i])    
        return res
    @classmethod
    def HistoryOfTotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        res = np.zeros(shape=len(positions))
        for i in range(len(positions)):
            res[i] = cls.TotalPotentialEnergy(masses=masses,positions=positions[i])    
        return res

    @classmethod
    def KineticEnergies(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(np.sum, axis=1, arr=(momenta ** 2))/(2 * masses)

    @classmethod
    def KineticEnergy(cls, masses: np.ndarray, momenta: np.ndarray) -> float:
        return cls.KineticEnergies(masses,momenta).sum()


    @classmethod
    def TotalPotentialEnergy(cls, masses: np.ndarray, positions: np.ndarray) -> float:
        result = 0
        for j in range(1,len(positions)):
            temp = 0
            for i in range(j):
                temp -= masses[i]/np.linalg.norm(positions[i]-positions[j], ord=2)

            temp *= masses[j] 
            result += temp
        return result
    
    @classmethod
    def dHdp(cls, masses: np.ndarray, momenta: np.ndarray) -> np.ndarray:
        return momenta / np.transpose(np.array([masses]))
    
    @classmethod
    def dHdq(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        # We calculate all the differences between the bodies
        pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        # transform them into distances
        distances = np.linalg.norm(pos_diff, axis=2)
        distances[distances == 0] = np.inf

        # obtaining directions
        directions = pos_diff / distances[:, :, np.newaxis]

        # we calculate the resulting values of dH/dq
        contributions = masses[:, np.newaxis] * directions / distances[:,:, np.newaxis]**2
        result = masses[:,np.newaxis]*contributions.sum(axis=1)
        
        return result

    @classmethod
    def dHdqDUMDUM(cls, masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        result = []
        for i in range(len(positions)):
            temp = 0
            for j in range(len(positions)):
                if j == i : continue
                
                temp -= (positions[i]-positions[j])* masses[j]/pow(np.linalg.norm(positions[i]-positions[j]),3)

            temp *= masses[i] 
            result.append(temp)
        return np.array(result)        


if __name__ == "__main__":
    a = NewtonHamiltonian
    masses = np.array([4,4,4,4,4])
    positions = np.array([np.array([0,0]),np.array([30,4]),np.array([5,-60]),np.array([7,80]),np.array([9,100])])
    momenta = np.array([np.array([1,2]),np.array([3,10]),np.array([5,90]),np.array([7,8]),np.array([9,10])])
    
  
    print("Kinetic energies")
    print(a.KineticEnergies(masses, momenta))
   
    print("Total kinetic energy")
    print(a.KineticEnergy(masses, momenta))
    
    print("Total potential energy")
    print(a.TotalPotentialEnergy(masses, positions))  
 
    print("dH / dp")
    print(a.dHdp(masses,momenta))
    
    print("dH / dq")
    print(a.dHdq(masses,positions))   

    print("dH / dq dumdum way")
    print(a.dHdqDUMDUM(masses,positions))
    

    ###############################################################
    ##  Performance testing
    ###############################################################
    test_times = True
 
    if test_times:
        import time

        samples = 1000

        masses = np.random.randint(1,1000,samples)
        positions = np.array([np.random.randint(1,100000,2) for i in range(samples)])
        momenta = np.array([np.random.randint(1,100000,2) for i in range(samples)])
    
        ### Test dHdp

        ti_dHdp = time.time()
        a.dHdp(masses,momenta)
        tf_dHdp = time.time()
        
        ### Test dHdq
        
        ti_dHdq = time.time()
        a.dHdq(masses,positions)
        tf_dHdq = time.time()
        
        ### Test dHdqDUMDUM 

        ti_dHdqDUMDUM = time.time()
        #a.dHdqDUMDUM(masses,positions)
        tf_dHdqDUMDUM = time.time()
        
        
        print(f"Runtimes on {samples} samples in 2D: \
            \n dHdp: {tf_dHdp-ti_dHdp} seconds\
            \n dHdq: {tf_dHdq-ti_dHdq} seconds\
            \n dHdqDUMDUM: {tf_dHdqDUMDUM-ti_dHdqDUMDUM} seconds")

        masses = np.random.randint(1,1000,samples)
        positions = np.array([np.random.randint(1,100000,3) for i in range(samples)])
        momenta = np.array([np.random.randint(1,100000,3) for i in range(samples)])
    
        ### Test dHdp

        ti_dHdp = time.time()
        a.dHdp(masses,momenta)
        tf_dHdp = time.time()
        
        ### Test dHdq
        ti_dHdq = time.time()
        a.dHdq(masses,positions)
        tf_dHdq = time.time()
        
        ### Test dHdqDUMDUM 
        ti_dHdqDUMDUM = time.time()
        #a.dHdqDUMDUM(masses,positions)
        tf_dHdqDUMDUM = time.time()
        
        
        print(f"Runtimes on {samples} samples in 3D: \
            \n dHdp: {tf_dHdp-ti_dHdp} seconds\
            \n dHdq: {tf_dHdq-ti_dHdq} seconds\
            \n dHdqDUMDUM: {tf_dHdqDUMDUM-ti_dHdqDUMDUM} seconds")
