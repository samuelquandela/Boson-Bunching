## Perceval 
import perceval as pcvl

## Other
import numpy as np
import matplotlib.pyplot as plt

def CreateVectors(n, w):
    #first two orthonormal vectors from which the Unitary matrix for the circuit is built
    v1 = np.zeros(n,dtype = 'complex_')
    v1[0] = 1
    v1[1] = 0
    for i in range(2,n):
        v1[i] = 1/np.sqrt(2)

    v2 = np.zeros(n,dtype = 'complex_')
    v2[0] = 0
    v2[1] = 1
    for i in range(2,n):
        v2[i] = np.conj(w**(i-2))/np.sqrt(2)

    return v1, v2

def normalize(vector):
    return vector / np.linalg.norm(vector)

def MakeUnitary(n, w):
    w = np.exp((2*np.pi*1j)/(n-2))
    vector1 = CreateVectors(n, w)[0]
    vector2 = CreateVectors(n, w)[1]

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

def CalcProb(disttype_input, n, Simulator):
    #initialize the probabilities
    Probability_photons = 0
    Probability_distribution_photons = []
    #summing over all cases where all photons end up in only two modes
    for i in range(n+1):
        p = Simulator.probability(disttype_input, pcvl.BasicState([i,n-i]+[0]* (n-2)))
        Probability_distribution_photons.append(p)
        Probability_photons += p
    for i in range(n+1):
        if Probability_photons != 0:
            Probability_distribution_photons[i] = Probability_distribution_photons[i]/Probability_photons

    return Probability_distribution_photons, Probability_photons

def Create_inputs(n,w):
#make a list of fully distinguishable photons of the form |{a:i}>
    theta = []
    for i in range(1,n+1):
        x = "|{{a:{}}}>".format(i)
        theta.append(pcvl.StateVector(x))
    #make a list of all input photons which are superpositions of |{a:1}> and |{a:2}>
    phi = [theta[0], theta[1]]
    for i in range(2,n):
        x = phi[0] + w**(i-2) * phi[1]
        phi.append(x)
    #Input states
    #initialise the variables
    x = 1 
    y = []
    z = 1
    #fill the states
    for i in range(n):
        y.append(1)
        x = x * phi[i]
        z = z * theta[i]
    return x , pcvl.BasicState(y), z

def addlabels(x,y):
    for i in range(len(x)):
        if y[i]>0.0005:
            plt.text(i,y[i]/2, round(y[i], 5), ha = 'center', color = 'white')
        else:
            plt.text(i,1.25*y[i], round(y[i], 5), ha = 'center')

def PlotProb(n, Probdist_partial , Prob_partial, Probdist_indist , Prob_indist, Probdist_fully , Prob_fully):
    colours= ['#58c4e1','#946cba','#383e48'] # light blue, purple and gray
    cases = ["indist", "partiallydist", "dist"]
    propabilities = [Prob_indist, Prob_partial, Prob_fully]
    plt.figure()
    plt.bar(cases, propabilities, color= colours)
    plt.ylabel('probability')
    plt.title("Comparing the total bunching probabilities for {} photons".format(n))
    addlabels(cases, propabilities)
    save_results_to = '/Users/samuelhorsch/Code/Perceval_Sam/BosonBunching/Figures/'
    plt.savefig(save_results_to + 'BunchingProbabilities{}photons.png'.format(n))
    #plt.savefig('Figures/BunchingProbabilities{}photons'.format(n))
    #plt.show()
    plt.close()
def PlotDist(n, Probdist_partial , Prob_partial, Probdist_indist , Prob_indist, Probdist_fully , Prob_fully):
    X = []
    for i in range(n+1):
        X.append("({},{})".format(i,n-i))
    outputmodes = X
    colours= ['#58c4e1','#946cba','#383e48'] # light blue, purple and gray
    plt.figure()
    plt.scatter(outputmodes,Probdist_fully, label= 'distinguishable', color = colours[0])
    plt.scatter(outputmodes,Probdist_partial, label = 'partially distinguishable', color = colours[1])
    plt.scatter(outputmodes,Probdist_indist, label = 'indistinguishable', color = colours[2])
    plt.ylabel('probability')
    plt.xlabel('number of photons in first two modes')
    plt.title("Normalised bunching distributions")
    plt.legend()
    save_results_to = '/Users/samuelhorsch/Code/Perceval_Sam/BosonBunching/Figures/'
    plt.savefig(save_results_to + 'BunchingDistribution{}photons.png'.format(n))
    #plt.savefig('Figures/BunchingDistribution{}photons'.format(n))
    #plt.show()
    plt.close()

        

def BosonBunching(n):
    #definition of w as the (n-2)th root of unity
    w = np.exp((2*np.pi*1j)/(n-2))
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

    input_partialydistinguishable = Create_inputs(n,w)[0] #partially distinguishable photons
    input_indisinguishable = Create_inputs(n,w)[1] #indistinguishable photons
    input_distinguishable = Create_inputs(n,w)[2] #fully distinguishable photons
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
    
    
for i in range(4,8):
    BosonBunching(i)
    

