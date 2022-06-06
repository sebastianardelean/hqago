#!/usr/bin/env python
# coding: utf-8

# In[453]:


import warnings; warnings.simplefilter('ignore')


# In[1]:


from qiskit import QuantumCircuit, execute, Aer, IBMQ,QuantumRegister,ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
#from qiskit.tools.jupyter import *
from qiskit.visualization import *
import networkx as nx

import operator


import math
import random
from qiskit import transpile, schedule as build_schedule
from qiskit.test.mock import FakeAlmaden

import pygad
import numpy as np
import csv

# In[2]:




# # Genetic Algorithm

# In[3]:


METAHEURISTIC_GENE_LENGTH:int = 4
METAHEURISTIC_GENE_LOW:int = -6
METAHEURISTIC_GENE_HIGH:int = 6
METAHEURISTIC_NUM_PARENS_MATING:int=2
METAHEURISTIC_POPULATION_SIZE:int = 100
METAHEURISTIC_SELECTION_TYPE:str="rws"
METAHEURISTIC_KEEP_PARENTS:int=1
METAHEURISTIC_CROSSOVER_RATE:float = 0.6
METAHEURISTIC_CROSSOVER_TYPE:str = "single_point"
METAHEURISTIC_MUTATION_TYPE:str = "adaptive" #"random"
METAHEURISTIC_MUTATION_RATE: float = 0.00002
METAHEURISTIC_NO_OF_GENERATIONS:int = 100


# mutation_probability
METAHEURISTIC_MUTATION_PROBABILITY = np.array([0.15, 0.005])

# mutation_num_genes
METAHEURISTIC_MUTATION_NUM_GENES = np.array([7, 2])

# mutation_percent_genes

METAHEURISTIC_MUTATION_PERCENT_GENES = np.array([21, 13])


# In[4]:


from collections import namedtuple
ItemConfiguration = namedtuple('Item', 'm,v')


# In[5]:


MAX_ALLOWED_MASS = 20
TOTAL_VALUE = 0
items = [ItemConfiguration(3, 3),
         ItemConfiguration(2, 5),
         ItemConfiguration(4, 10),
         ItemConfiguration(7, 4),
         ItemConfiguration(9, 15),
         
        
        ]
for item in items:
    TOTAL_VALUE+= item.v


desired_output = 33


# In[6]:
def check_duplicates(input_list):
    seen = set()
    uniq = []
    for el in input_list:
        if abs(el) not in seen:
            uniq.append(el)
            seen.add(abs(el))
    return uniq

def fitness_function(mass,value):
    return int(value - (TOTAL_VALUE+1)*divmod(mass,MAX_ALLOWED_MASS)[0])

#Accepts a function that must accept 2 parameters (a single solution and its index in the population) and return the fitness value of the solution
def fitness_func(solution, solution_idx):
    #print(solution)
    calculated_mass = 0
    calculated_value = 0
    output = 0
    if 0 in solution:
        return -1
    unique_elements = check_duplicates(solution)
    if len(unique_elements)<len(solution):
        return -2

    for gene in solution:
        position = abs(gene)-1
        if gene>0:
            calculated_mass += items[position].m
            calculated_value += items[position].v
    output = fitness_function(calculated_mass,calculated_value)
    if calculated_mass > MAX_ALLOWED_MASS:
        output = -3
    return output


# In[7]:


def run_ga():
  ga_instance = pygad.GA(num_generations=METAHEURISTIC_NO_OF_GENERATIONS,
                       num_parents_mating=METAHEURISTIC_NUM_PARENS_MATING,
                       gene_type = int,
                       fitness_func=fitness_func,
                       sol_per_pop=METAHEURISTIC_POPULATION_SIZE,
                       num_genes=METAHEURISTIC_GENE_LENGTH,
                       init_range_low=METAHEURISTIC_GENE_LOW,
                       init_range_high=METAHEURISTIC_GENE_HIGH,
                       parent_selection_type=METAHEURISTIC_SELECTION_TYPE,
                       keep_parents=METAHEURISTIC_KEEP_PARENTS,
                       crossover_type=METAHEURISTIC_CROSSOVER_TYPE,
                       crossover_probability = METAHEURISTIC_CROSSOVER_RATE,
                       mutation_type=METAHEURISTIC_MUTATION_TYPE,
#                       mutation_probability = METAHEURISTIC_MUTATION_RATE,
#                       mutation_probability = METAHEURISTIC_MUTATION_PROBABILITY,
                       mutation_percent_genes = METAHEURISTIC_MUTATION_PERCENT_GENES,
                       stop_criteria="saturate_10"
                       )
  ga_instance.run()
  solution, solution_fitness, solution_idx = ga_instance.best_solution()
  #print("Parameters of the best solution : {solution}".format(solution=solution))
  #print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
  #print("Number of generations={generations}".format(generations=ga_instance.best_solution_generation))
  return solution.tolist(), ga_instance.generations_completed


# # RQGA

# In[8]:


NO_OF_QUBITS_FITNESS = 8
NO_OF_QUBITS_INDIVIDUAL = 5
POPULATION_SIZE = 2**NO_OF_QUBITS_INDIVIDUAL
NO_OF_MAX_GROVER_ITERATIONS = int(math.sqrt(2**NO_OF_QUBITS_INDIVIDUAL))-1
NO_OF_QUBITS_CARRY = 2 * NO_OF_MAX_GROVER_ITERATIONS + 1


# In[9]:


def to_binary(value, number_of_bits, lsb=False):
    """
    Function return two's complement representation
    :param value: value in decimal representation
    :param number_of_bits: number of bits used for representation
    :returns: np.array that represents the binary representation
    >>> to_binary(10,4)
    array([1, 0, 1, 0])
    >>> to_binary(10,4,True)
    array([0, 1, 0, 1])
    """
    if lsb == True:
        return np.flip(np.array(list(np.binary_repr(value, number_of_bits)), dtype=int))
    return np.array(list(np.binary_repr(value, number_of_bits)), dtype=int)


# In[10]:


def get_random_value():
    # 2 ^ (M+1) < max < 2 ^ (M+2)
    return random.randrange(0,2**(NO_OF_QUBITS_FITNESS))


def get_marked_individuals(individual_mho, individual_binary):
    marked_individual=[]
    is_marked = False
    if METAHEURISTIC_GENE_LENGTH == 0:
        for value in individual_binary:
            marked_individual.append((value,True))
        is_marked=True
    else:
        for value in individual_binary:
            marked_individual.append((value,False))
        for gene in individual_mho:
            position = abs(gene)-1
            if gene>0:
                if individual_binary[position] == 1:
                    value = marked_individual[position]
                    new_value=(value[0],True)
                    marked_individual[position]=new_value
            else:
                if individual_binary[position] == 0:
                    value = marked_individual[position]
                    new_value=(value[0],True)
                    marked_individual[position]=new_value
        
        marked_only_size = len(list(filter(lambda x:(x[1]==True), marked_individual)))
        if marked_only_size == len(individual_mho):
            is_marked = True
    return marked_individual,is_marked
# In[11]:


def calculate_individual_fitness(individual_bin_configuration):
    calculated_mass = 0
    calculated_value = 0
    for i in range(0, NO_OF_QUBITS_INDIVIDUAL):
        if individual_bin_configuration[i]==1:
            calculated_mass += items[i].m
            calculated_value += items[i].v
    fitness_value = fitness_function(calculated_mass,calculated_value)
    if calculated_mass < MAX_ALLOWED_MASS:
        return True, fitness_value
    else:
        return False,fitness_value


def recreate_individual(individual_mho, individual_binary):
    solution = [(0,False) for _ in range(0,NO_OF_QUBITS_INDIVIDUAL)]
    if METAHEURISTIC_GENE_LENGTH == 0:
        return individual_binary
    for gene in individual_mho:
        position=abs(gene)-1
        if gene>0:
            new_value = (1,True)
            solution[position]=new_value
        else:
            new_value = (0, True)
            solution[position]=new_value
    for gene in individual_binary:
        for i in range(0,len(solution)):
            value = solution[i]
            if value[1]==False:
                new_value = (gene,True)
                solution[i]=new_value
                break
    return list(map(lambda x: x[0], solution))
# In[12]:


def get_ufit_instruction(individual_mho):
    #define and initialize the individual quantum register
    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL-METAHEURISTIC_GENE_LENGTH,"ind_qreg")
    #define and initialize the fitness quantum register. 
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fit_qreg")
    #create the ufit subcircuit
    qc = QuantumCircuit(ind_qreg,fit_qreg,name="U$_fit$")
    for i in range(0,POPULATION_SIZE):
        """
        For each individual in population get the two's complement representation and 
        set the qubits on 1 using X-gate, according to the binary representation
        """
        individual_binary = to_binary(i, NO_OF_QUBITS_INDIVIDUAL, True)
        marked_individual,is_marked=get_marked_individuals(individual_mho,individual_binary)
        if not is_marked:
            continue            
        qubit_index=0
        for value in marked_individual:
            if value[0]==0 and value[1]==False:
                qc.x(ind_qreg[qubit_index])
                qubit_index+=1
        """
        Calculate the fitness value and get the two's complement representation of the fitness value.
        """        
        new_individual_binary = list(map(lambda x : x[0], marked_individual))
        valid,fitness_value = calculate_individual_fitness(new_individual_binary)
        fitness_value_binary = to_binary(fitness_value,NO_OF_QUBITS_FITNESS,True)
        
        

        """
        Set the fitness value in fitness quantum register for each individual and mark it valid or invalid
        """
        for k in range(0,NO_OF_QUBITS_FITNESS):
            if fitness_value_binary[k]==1:
                qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL-METAHEURISTIC_GENE_LENGTH)],fit_qreg[k])
        #if fitness value si greater than 0 then set the valid qubit to 1
        if valid == True:
            qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL-METAHEURISTIC_GENE_LENGTH)],fit_qreg[NO_OF_QUBITS_FITNESS])
        
        
        #reset individual
        qubit_index=0
        for value in marked_individual:
            if value[0]==0 and value[1]==False:
                qc.x(ind_qreg[qubit_index])
                qubit_index+=1
        qc.barrier()
    return qc.to_instruction()


# In[13]:


def get_oracle_instruction(positive_value_array):
    #define and initialize fitness quantum register
    fit_reg = QuantumRegister(NO_OF_QUBITS_FITNESS,"fqreg")
    #define and initialize max quantum register
    no_of_edges_reg=QuantumRegister(NO_OF_QUBITS_FITNESS,"noqreg")
    #define and initialize carry quantum register
    carry_reg = QuantumRegister(3,"cqreg")
    #define and initialize oracle workspace quantum register
    oracle = QuantumRegister(1,"oqreg")
    #create Oracle subcircuit
    oracle_circ = QuantumCircuit(fit_reg,no_of_edges_reg,carry_reg,oracle,name="O")
    
    #define majority operator
    def majority(circ,a,b,c):
        circ.cx(c,b)
        circ.cx(c,a)
        circ.ccx(a, b, c)
    #define unmajority operator
    def unmaj(circ,a,b,c):
        circ.ccx(a, b, c)
        circ.cx(c, a)
        circ.cx(a, b)
    #define the Quantum Ripple Carry Adder
    def adder_8_qubits(p,a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7,cin,cout):
        majority(p, cin, b0, a0)
        majority(p, a0, b1, a1)
        majority(p, a1, b2, a2)
        majority(p, a2, b3, a3)
        majority(p, a3, b4, a4)
        majority(p, a4, b5, a5)
        majority(p, a5, b6, a6)
        majority(p, a6, b7, a7)
        
        p.cx(a7, cout)
        
        unmaj(p, a6, b7, a7)
        unmaj(p, a5, b6, a6)
        unmaj(p, a4, b5, a5)
        unmaj(p, a3, b4, a4)
        unmaj(p, a2, b3, a3)
        unmaj(p, a1, b2, a2)
        unmaj(p, a0, b1, a1)
        unmaj(p, cin, b0, a0)
    
    """
    Subtract max value. We start by storing the max value in the quantum register. Such, considering that 
    all qubits are |0>, if on position i in positive_value_array there's 0, then qubit i will be negated. Otherwise, 
    if on position i in positive_value_array there's a 1, by default will remain 0 in no_of_edges_reg quantum
    register. For performing subtraction, carry-in will be set to 1.
    """
    for i in range(0,NO_OF_QUBITS_FITNESS):
        if positive_value_array[i]==0:
            oracle_circ.x(no_of_edges_reg[i])
    oracle_circ.x(carry_reg[0])

    adder_8_qubits(oracle_circ, 
            no_of_edges_reg[0],no_of_edges_reg[1],no_of_edges_reg[2],no_of_edges_reg[3],
            no_of_edges_reg[4],no_of_edges_reg[5],no_of_edges_reg[6],no_of_edges_reg[7],       
            fit_reg[0],fit_reg[1],fit_reg[2],fit_reg[3],
            fit_reg[4],fit_reg[5],fit_reg[6],fit_reg[7],
               carry_reg[0],carry_reg[1]);

    
    oracle_circ.barrier()
    """
    Reset the value in no_of_edges_reg and carry-in
    """
    oracle_circ.x(no_of_edges_reg)
    oracle_circ.x(carry_reg[0])
    
    """
    Mark the corresponding basis states by shifting their amplitudes.
    """
    
    oracle_circ.h(oracle[0])
    oracle_circ.mct([fit_reg[i] for i in range(0,NO_OF_QUBITS_FITNESS)],oracle[0])
    oracle_circ.h(oracle[0])
    
    """
    Restore the fitness value by adding max value.
    """
    adder_8_qubits(oracle_circ, 
            no_of_edges_reg[0],no_of_edges_reg[1],no_of_edges_reg[2],no_of_edges_reg[3],
            no_of_edges_reg[4],no_of_edges_reg[5],no_of_edges_reg[6],no_of_edges_reg[7],       
            fit_reg[0],fit_reg[1],fit_reg[2],fit_reg[3],
            fit_reg[4],fit_reg[5],fit_reg[6],fit_reg[7],
            carry_reg[0],carry_reg[2]);
    return oracle_circ.to_instruction()


# In[14]:


def get_grover_iteration_subcircuit():
    #define and initialize fitness quantum register
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fqreg")
    #define and initialize oracle workspace quantum register
    oracle_ws = QuantumRegister(1,"ows")
    #create grover diffuser subcircuit
    grover_circ = QuantumCircuit(fit_qreg,oracle_ws,name ="U$_s$")

    grover_circ.h(fit_qreg)
    grover_circ.x(fit_qreg)

    grover_circ.h(oracle_ws[0])

    grover_circ.mct(list(range(NO_OF_QUBITS_FITNESS+1)), oracle_ws[0])  # multi-controlled-toffoli

    grover_circ.h(oracle_ws[0])


    grover_circ.x(fit_qreg)
    grover_circ.h(fit_qreg)
    grover_circ.h(oracle_ws)

    return grover_circ.to_instruction()


# In[15]:


def string_to_list(string):
    return_list = []
    for i in string:
        if i != " ":
            return_list.append(int(i))
    return return_list  


# In[16]:


def run_algorithm(run_no, writer):
    #Load IBMQ account
    IBMQ.load_account()
    #calculate the number of edges in graph
    max_value = 0
    new_fitness_value = get_random_value()
    print("Max value:{0}".format(new_fitness_value))
    #define a list for storing the results
    final_results = []
    individual_mho=[]
    number_of_ga_generations = 0

    print("Preparing quantum registers and creating quantum circuit...")
    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL-METAHEURISTIC_GENE_LENGTH,"ireg")
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"freg")
    carry_qreg = QuantumRegister(NO_OF_QUBITS_CARRY,"qcarry")
    oracle = QuantumRegister(1,"oracle")
    creg = ClassicalRegister(NO_OF_QUBITS_INDIVIDUAL-METAHEURISTIC_GENE_LENGTH,"reg")
    max_value_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS,"max_value_qreg")

    print("Creating quantum circuit...")

    qc = QuantumCircuit(ind_qreg,fit_qreg,carry_qreg,oracle,max_value_qreg,creg)

    

    print("Metaheuristic...")
    if METAHEURISTIC_GENE_LENGTH == NO_OF_QUBITS_INDIVIDUAL:
        individual_mho, number_of_ga_generations = run_ga()
        writer.writerow({"algo_run_no":run_no,"solution":individual_mho, "ga_generation":number_of_ga_generations})
        return;
        
    elif METAHEURISTIC_GENE_LENGTH == 0:
        #pure quantum, no need to call run_ga
        individual_mho=[]
    else:
        individual_mho,number_of_ga_generations = run_ga()
    
    
    print("Creating superposition of individuals...")
    qc.h(ind_qreg)
    qc.h(oracle)
    


    print("Getting ufit, oracle and grover iterations subcircuits...")
    ufit_instr = get_ufit_instruction(individual_mho)
  

    print("Append Ufit instruction to circuit...")
    qc.append(ufit_instr, [ind_qreg[q] for q in range(0,NO_OF_QUBITS_INDIVIDUAL-METAHEURISTIC_GENE_LENGTH)]+
                          [fit_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS+1)]
            )
    generation = 0
    while new_fitness_value != max_value and generation < 8:
        max_value = new_fitness_value
        max_value_bin = to_binary(max_value,NO_OF_QUBITS_FITNESS,True)
        oracle_instr = get_oracle_instruction(max_value_bin)
        grover_iter_inst = get_grover_iteration_subcircuit()
        for it in range(0,NO_OF_MAX_GROVER_ITERATIONS):
            print("Append Oracle instruction to circuit...")

            qc.append(oracle_instr,[fit_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS)]+
                               [max_value_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS)]+
                               [carry_qreg[0],carry_qreg[2*it+1],carry_qreg[2*it+2]]+
                               [oracle[0]]
                     )
            print("Append Grover Diffuser to circuit...")
            qc.append(grover_iter_inst, [fit_qreg[q] for q in range(0,NO_OF_QUBITS_FITNESS+1)]+
                                        [oracle[0]]
                     )

        print("Measure circuit...")
        qc.measure(ind_qreg,creg)

        simulation_results = []

        provider = IBMQ.get_provider(hub='ibm-q',group='open', project='main')
        backend = provider.get_backend('simulator_mps')
        # Perform 10 measurements for each circuit
       
        print("Setup simulator...")    
        shots = 16
        try:
            print("Starting simulator...")
            mapped_circuit = transpile(qc, backend=backend)
            qobj = assemble(mapped_circuit, backend=backend, shots=shots)
            runner = backend.run(qobj)
            job_monitor(runner)
            results = runner.result()
            answer = results.get_counts()
            #Get the result with the maximum number of counts
            max_item =max(answer.items(), key=operator.itemgetter(1))
            solution_individual=string_to_list(max_item[0])
            solution_individual=recreate_individual(individual_mho,solution_individual)
            _,new_fitness_value=calculate_individual_fitness(solution_individual)
            new_fitness_value = abs(new_fitness_value)
            writer.writerow({"algo_run_no":run_no,"solution":solution_individual, "fitness_value":new_fitness_value,"rqga_generation":generation,"ga_generation":number_of_ga_generations})

            print("Found solution {0} with fitness {1}...".format(solution_individual,new_fitness_value))
        except Exception as e:
            print(str(e))
        generation+=1
        qc.reset(max_value_qreg)


# In[17]:


from tqdm import tqdm
import sys

import contextlib


with open("results.csv", "w", newline='') as csvfile:
    fieldnames = ["algo_run_no", "solution", "fitness_value","rqga_generation", "ga_generation"]
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()

    for i in tqdm(range(100)):
        run_algorithm(i,writer)


# In[ ]:



