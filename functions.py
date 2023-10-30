## Perceval 
import perceval as pcvl

## Other
import numpy as np
import math
import matplotlib.pyplot as plt

def CreateVectors(n,w):
    #first two orthonormal vectors from which the Unitary matrix for the circuit is built
    v1 = np.zeros(n,dtype = 'complex_')
    v1[0] = 1
    v1[1] = 0
    for i in range(2,n):
        v1[i] = 1/math.sqrt(2)

    v2 = np.zeros(n,dtype = 'complex_')
    v2[0] = 0
    v2[1] = 1
    for i in range(2,n):
        v2[i] = np.conj(w**(i-2))/math.sqrt(2)

    return v1, v2 # these are the complex conjugate of the first two rows of the matrix M for n=7

def normalize(vector):
    return vector / np.linalg.norm(vector)

def MakeUnitary(n, w):
    vector1, vector2 = CreateVectors(n,w)

    orthonormal_basis = [normalize(vector1), normalize(vector2)]
    for _ in range(n-2):
        random_vector = np.random.rand(n) +  1j*np.random.rand(n) # Create a random 7-dimensional complex vector
        # Subtract projections onto existing basis vectors to make it orthogonal
        for basis_vector in orthonormal_basis:
            random_vector -= np.vdot(basis_vector,random_vector) * basis_vector
            # Normalize the orthogonal vector to make it orthonormal
        orthonormal_basis.append(normalize(random_vector))
        # Define the unitary matrix from the calculated vectors
    mat = []
    for i in range(n):
        mat.append(orthonormal_basis[i])
    UnitaryMatrix = np.matrix(mat)

    return UnitaryMatrix

def Create_inputs(n,w):
    #make a list of fully distinguishable photons of the form |{a:i}> ## I would define what |{a:i}> means i.e. attribute a of the photon has value i
    ##For a list of states, I'd rather choose another name than theta
    States_dist = []
    for i in range(1,n+1):
        x = "|{{a:{}}}>".format(i)
        States_dist.append(pcvl.StateVector(x))
    #make a list of all input photons which are superpositions of |{a:1}> and |{a:2}>
    States_par_dist = [States_dist[0], States_dist[1]] #start with the pure states |{a:1}> and |{a:2}>
    for i in range(2,n):
        x = States_par_dist[0] + w**(i-2) * States_par_dist[1] #add the states |{a:1}> + w^(i-2)|{a:2}>
        States_par_dist.append(x)
    #Input states
    #initialise the variables
    indisting_photons = [] 
    par_disting_photons = 1 
    disting_photons = 1 
    #fill the states
    for i in range(n):
        indisting_photons.append(1) # gives the state |1,1, ... , 1>
        par_disting_photons = par_disting_photons * States_par_dist[i]
        disting_photons = disting_photons * States_dist[i] # gives the state |{a:1},{a:2}, ... ,{a:n}> 
    return pcvl.BasicState(indisting_photons), par_disting_photons, disting_photons

def CalcProb(disttype_input, n, Simulator):
    def Reverse(lst):
        new_lst = lst[::-1]
        return new_lst
    #initialize the probabilities
    Probability_photons = 0
    Probability_distribution_photons = []
    #summing over all cases where all photons end up in only two modes
    for i in range(math.ceil((n+1)/2)):  
        p = Simulator.probability(disttype_input, pcvl.BasicState([i,n-i]+[0]* (n-2)))
        Probability_distribution_photons.append(p)
        Probability_photons += p
    X = []
    for i in range(math.ceil((n+1)/2), n+1):
        X.append(Probability_distribution_photons[i-math.ceil((n+1)/2)])
        Probability_photons += Probability_distribution_photons[i-math.ceil((n+1)/2)]
    Probability_distribution_photons = Probability_distribution_photons + Reverse(X)
    for i in range(n+1):
        if Probability_photons != 0:
            Probability_distribution_photons[i] = Probability_distribution_photons[i]/Probability_photons
    
    return Probability_distribution_photons, Probability_photons

colours= ['#58c4e1','#946cba','#383e48'] # light blue, purple and gray
def PlotProb(n, Probdist_partial , Prob_partial, Probdist_indist , Prob_indist, Probdist_fully , Prob_fully):
    def addlabels(x,y):
        for i in range(len(x)):
            if y[i]>0.0005:
                plt.text(i,y[i]/2, round(y[i], 5), ha = 'center', color= 'white')
            else:
                plt.text(i,1.25*y[i], round(y[i], 5), ha = 'center')
    
    cases = ["indist", "partiallydist", "dist"]
    propabilities = [Prob_indist, Prob_partial, Prob_fully]
    plt.figure()
    plt.bar(cases, propabilities, color= colours)
    plt.ylabel('probability')
    plt.title("Comparing the total bunching probabilities for {} photons".format(n))
    addlabels(cases, propabilities)
    save_results_to = '/Users/samuelhorsch/Code/Perceval_Sam/BosonBunching/Figures/'
    plt.savefig(save_results_to + 'BunchingProbabilities{}photons.png'.format(n))
    plt.close()

def PlotDist(n, Probdist_partial , Prob_partial, Probdist_indist , Prob_indist, Probdist_fully , Prob_fully):
    X = []
    for i in range(n+1):
        X.append("({},{})".format(i,n-i)) #creates a list of all posible tuple such that a+b=n for tuple (a,b)
    outputmodes = X

    plt.figure()
    plt.scatter(outputmodes,Probdist_indist, label = 'indistinguishable', color = colours[0])
    plt.scatter(outputmodes,Probdist_partial, label = 'partially distinguishable', color= colours[1])
    plt.scatter(outputmodes,Probdist_fully, label= 'distinguishable', color= colours[2])
    plt.ylabel('probability')
    plt.xlabel('number of photons in first two modes')
    plt.title("Normalised bunching distributions")
    plt.legend()
    save_results_to = '/Users/samuelhorsch/Code/Perceval_Sam/BosonBunching/Figures/'
    plt.savefig(save_results_to + 'BunchingDistribution{}photons.png'.format(n))
    plt.close()

def BosonBunching(n):
    #definition of w as the (n-2)th root of unity
    w = np.exp((2*math.pi*1j)/(n-2))
    UnitaryMatrix = MakeUnitary(n, w)
    # building Mach-Zender Interferometer block of the circuit
    mzi = (pcvl.BS() // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
        // pcvl.BS() // (1, pcvl.PS(phi=pcvl.Parameter("φ_b"))))
    # convert Unitary matrix into perceval languange
    Unitary = pcvl.Matrix(UnitaryMatrix)
    #create circuit
    Circuit_Rand = pcvl.Circuit.decomposition(Unitary, mzi,
                                                phase_shifter_fn=pcvl.PS,
                                                shape="triangle")
    #pcvl.pdisplay(Circuit_Rand, recursive=True)

    input_indisinguishable, input_partialydistinguishable, input_distinguishable = Create_inputs(n,w) #indistinguishable photons, partially distinguishable photons, fully distinguishable photons respectivley

    #simulating boson sampling
    p = pcvl.Processor("SLOS")
    p.set_circuit(Circuit_Rand)
    s = pcvl.SimulatorFactory().build(p)

    Probdist_partial , Prob_partial = CalcProb(input_partialydistinguishable, n, s) # high runtime
    Probdist_indist , Prob_indist = CalcProb(input_indisinguishable, n, s)
    Probdist_fully , Prob_fully = CalcProb(input_distinguishable, n, s) # high runtime

    print("The probability for all {} {} photon ending up in the first two modes is: {:0.3f} %".format(n,"disinguishable",Prob_fully*100))
    print("The probability for all {} {} photon ending up in the first two modes is: {:0.3f} %".format(n,"indisinguishable",Prob_indist*100))
    print("The probability for all {} {} photon ending up in the first two modes is: {:0.3f} %".format(n,"parially disinguishable",Prob_partial*100))

    PlotProb(i, Probdist_partial , Prob_partial, Probdist_indist , Prob_indist, Probdist_fully , Prob_fully)
    PlotDist(i, Probdist_partial , Prob_partial, Probdist_indist , Prob_indist, Probdist_fully , Prob_fully)
    
    
# for i in range(4,8):
#     BosonBunching(i)
    
BosonBunching(8)
