## Perceval 
import perceval as pcvl

## Other
import numpy as np
import math
import matplotlib.pyplot as plt

def create_vectors(n,w):
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

def make_unitary(n, w):
    vector1, vector2 = create_vectors(n,w)

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
    unitary_matrix = np.matrix(mat)

    return unitary_matrix

def create_inputs(n,w):
    #make a list of fully distinguishable photons of the form |{a:i}> that means we give the photon an attribute a with value i 
    states_dist = []
    for i in range(1,n+1):
        x = "|{{a:{}}}>".format(i)
        states_dist.append(pcvl.StateVector(x))
    #make a list of all input photons which are superpositions of |{a:1}> and |{a:2}>
    states_par_dist = [states_dist[0], states_dist[1]] #start with the pure states |{a:1}> and |{a:2}>
    for i in range(2,n):
        x = states_par_dist[0] +  states_par_dist[1] * w**(i-2) #add the states |{a:1}> + w^(i-2)|{a:2}>
        states_par_dist.append(x)
    #Input states
    #initialise the variables
    indisting_photons = [] 
    par_disting_photons = 1 
    disting_photons = 1 
    #fill the states
    for i in range(n):
        indisting_photons.append(1) # gives the state |1,1, ... , 1>
        par_disting_photons = par_disting_photons * states_par_dist[i]
        disting_photons = disting_photons * states_dist[i] # gives the state |{a:1},{a:2}, ... ,{a:n}> 
    return pcvl.BasicState(indisting_photons), par_disting_photons, disting_photons

def calc_prob(disttype_input, n, simulator):
    def reverse(lst):
        new_lst = lst[::-1]
        return new_lst
    #initialize the probabilities
    probability_photons = 0
    probability_distribution_photons = []
    #summing over all cases where all photons end up in only two modes
    for i in range(math.ceil((n+1)/2)): # range over half the distribution because of symmetry
        p = simulator.probability(disttype_input, pcvl.BasicState([i,n-i]+[0]* (n-2)))
        probability_distribution_photons.append(p)
        probability_photons += p
    X = []
    for i in range(math.ceil((n+1)/2), n+1):
        X.append(probability_distribution_photons[i-math.ceil((n+1)/2)])
        probability_photons += probability_distribution_photons[i-math.ceil((n+1)/2)]
    probability_distribution_photons = probability_distribution_photons + reverse(X)
    for i in range(n+1):
        if probability_photons != 0:
            probability_distribution_photons[i] = probability_distribution_photons[i]/probability_photons
    
    return probability_distribution_photons, probability_photons

colours= ['#58c4e1','#946cba','#383e48'] # light blue, purple and gray
def plot_prob(n, probdist_partial , prob_partial, probdist_indist , prob_indist, probdist_fully , prob_fully):
    def add_labels(x,y):
        for i in range(len(x)):
            if y[i]>0.0005:
                plt.text(i,y[i]/2, round(y[i], 5), ha = 'center', color= 'white')
            else:
                plt.text(i,1.25*y[i], round(y[i], 5), ha = 'center')
    
    cases = ["indist", "partiallydist", "dist"]
    propabilities = [prob_indist, prob_partial, prob_fully]
    plt.figure()
    plt.bar(cases, propabilities, color= colours)
    plt.ylabel('probability')
    plt.title("Comparing the total bunching probabilities for {} photons".format(n))
    add_labels(cases, propabilities)
    save_results_to = '/Users/samuelhorsch/Code/Perceval_Sam/BosonBunching/Figures/'
    plt.savefig(save_results_to + 'BunchingProbabilities{}photons.png'.format(n))
    plt.close()

def plot_dist(n, probdist_partial , prob_partial, probdist_indist , prob_indist, probdist_fully , prob_fully):
    X = []
    for i in range(n+1):
        X.append("({},{})".format(i,n-i)) #creates a list of all posible tuple such that a+b=n for tuple (a,b)
    outputmodes = X

    plt.figure()
    plt.scatter(outputmodes,probdist_indist, label = 'indistinguishable', color = colours[0])
    plt.scatter(outputmodes,probdist_partial, label = 'partially distinguishable', color= colours[1])
    plt.scatter(outputmodes,probdist_fully, label= 'distinguishable', color= colours[2])
    plt.ylabel('probability')
    plt.xlabel('number of photons in first two modes')
    plt.title("Normalised bunching distributions")
    plt.legend()
    save_results_to = '/Users/samuelhorsch/Code/Perceval_Sam/BosonBunching/Figures/'
    plt.savefig(save_results_to + 'BunchingDistribution{}photons.png'.format(n))
    plt.close()

def boson_bunching(n):
    #definition of w as the (n-2)th root of unity
    w = np.exp((2*math.pi*1j)/(n-2))
    unitary_matrix = make_unitary(n, w)
    # building Mach-Zender Interferometer block of the circuit
    mzi = (pcvl.BS() // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
        // pcvl.BS() // (1, pcvl.PS(phi=pcvl.Parameter("φ_b"))))
    # convert Unitary matrix into perceval language
    unitary = pcvl.Matrix(unitary_matrix)
    #create circuit
    circuit_rand = pcvl.Circuit.decomposition(unitary, mzi,
                                                phase_shifter_fn=pcvl.PS,
                                                shape="triangle")
    

    input_indistinguishable, input_partially_distinguishable, input_distinguishable = create_inputs(n,w) #indistinguishable photons, partially distinguishable photons, fully distinguishable photons respectivley

    #simulating boson sampling
    p = pcvl.Processor("SLOS")
    p.set_circuit(circuit_rand)
    s = pcvl.SimulatorFactory().build(p)

    probdist_partial , prob_partial = calc_prob(input_partially_distinguishable, n, s) # high runtime
    probdist_indist , prob_indist = calc_prob(input_indistinguishable, n, s)
    probdist_fully , prob_fully = calc_prob(input_distinguishable, n, s) # high runtime

    print("The probability for all {} {} photon ending up in the first two modes is: {:0.3f} %".format(n,"distinguishable",prob_fully*100))
    print("The probability for all {} {} photon ending up in the first two modes is: {:0.3f} %".format(n,"indistinguishable",prob_indist*100))
    print("The probability for all {} {} photon ending up in the first two modes is: {:0.3f} %".format(n,"partially distinguishable",prob_partial*100))

    plot_prob(n, probdist_partial , prob_partial, probdist_indist , prob_indist, probdist_fully , prob_fully)
    plot_dist(n, probdist_partial , prob_partial, probdist_indist , prob_indist, probdist_fully , prob_fully)
    
       
boson_bunching(5)
