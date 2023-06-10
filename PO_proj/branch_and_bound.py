from mip import *
from collections import  namedtuple
import sys
import matplotlib.pyplot as plt


Instance = namedtuple("Instance", ['model', 'variables', 'n_var', 'n_cons'])
epsilon = sys.float_info.epsilon * 30



def read_instance(path:str) -> Instance:
    model = Model(sense= MAXIMIZE, solver_name= CBC)
    model.verbose = 0
    
    file = open(path, 'r')
    lines = file.readlines()

    n_var, n_cons = [int(x) for x in lines[0].split()] # numero de variaveis e restricoes

    variables = [model.add_var(var_type= CONTINUOUS, name = f'x{i + 1}', lb = 0.0, ub = 1.0) for i in range(n_var)]
    # as variáveis são contínuas para que possam assumir valores fracionários no Branch and Bound

    coefficients = [float(x) for x in lines[1].split()] # coeficientes da FO

    model.objective = xsum(coefficients[i] * variables[i] for i in range(n_var))

    for i in range(2, len(lines)):
        constrict_coefficients = [float(x) for x in lines[i].split()]  # coeficientes da restricao
                                                                       # pela regra, os n_var primeiros coeficientes
                                                                       # corresponderão às variáveis e o último ao  RHS

        model += xsum(constrict_coefficients[j] * variables[j] for j in range(n_var)) <= constrict_coefficients[-1]



    return Instance(model, variables, n_var, n_cons)

class Node:
    def __init__(self, model:mip.Model, variables: mip.VarList, level:int = 0, left = None, right = None):
        self.model = model
        self.variables = variables
        self.level = level
        self.left = left
        self.right = right
        self.is_integer = False

        #self.is_open = True
        
    
    def __str__(self) -> str:
        representation = f'Objective Value: {self.model.objective_value}\n'
        representation += f'Variables: {[x.x for x in self.variables]}\n'
        representation += f'Is integer: {self.is_integer}\n'
        representation += f'Level: {self.level}\n'
      
        return representation
    
    def __repr__(self) -> str:
        return self.__str__()


def is_integer_variable(v: mip.Var) -> bool:
    if (v.x >= 0.5) and (v.x + epsilon < 1):    # se x + epsilon for menor que 1, então x é fracionário
        return False   
    elif (v.x < 0.5) and (v.x - epsilon > 0):
        return False                               # se x - epsilon for maior que 0, então x é fracionário
    return True

def is_integer_set(variables:mip.VarList) -> bool:
    for x in variables:
        if is_integer_variable(x) == False:
            return False
    return True


def get_best_node(left:Node, right:Node) -> Node:
    
    if (left.model.objective_value - right.model.objective_value) > epsilon:
           return left
    else:
          return right


def plot_tree(node:Node, ax:plt.axes, base:int = 2, x = 0, y = 0 , parent_x = None, parent_y = None):
  
    ax.annotate(f'{node.level}\n{node.model.objective_value:.2f}\n{node.is_integer}\n', (x, y), ha='center', va='center', bbox=dict(boxstyle='circle', fc='w'))
    if parent_x != None:
        ax.plot([parent_x, x], [parent_y, y], color='k', linewidth=1.0, linestyle='-')
    if node.left:
        plot_tree(node.left, ax, base, x - (base) ** (10 - node.level), y - 1, x, y)
    if node.right:
        plot_tree(node.right, ax,base,  x + (base) ** (10 - node.level), y - 1, x, y)


    

class BranchAndBound(object):
    __bb_key = object()
    def __init__(self, root:Node, key = None):
        try:
            assert key == BranchAndBound.__bb_key
        except AssertionError:
            raise ValueError('BranchAndBound must be instantiated using BranchAndBound.from_instance()')
        self.root = root
   
    
    @classmethod
    def from_instance(cls, instance:Instance):
        
        BranchAndBound.primal_limit = float('-inf')
        BranchAndBound.best_node = None
        BranchAndBound.finished = False
        BranchAndBound.initial = Node(instance.model, instance.variables) 
        return cls(BranchAndBound.initial, BranchAndBound.__bb_key)

    def __str__(self) -> str:
        if BranchAndBound.finished:
            representation = f'Primal limit: {BranchAndBound.primal_limit}\n'
            
            representation += f'Best Node: {BranchAndBound.best_node}\n'
        else:
            representation = 'Branch and Bound is not optimized yet'
        return representation
    

    def __repr__(self) -> str:
        return self.__str__()
    

    def optimize(self):
        self.__solve()
        BranchAndBound.finished = True
        
        return BranchAndBound.best_node
    
       
    @classmethod
    def plot_tree(cls, base:int = 2, size_x:int = 20, size_y:int = 12, set_y_visible:bool = False):
        if BranchAndBound.finished == False:
            raise ValueError('Branch and Bound is not optimized yet')
        ax = plt.gca()
        ax.figure.set_size_inches(size_x, size_y)
        ax.get_xaxis().set_visible(False)
        if set_y_visible == False:
            ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plot_tree(BranchAndBound.initial, ax = ax, base = base)


    

    def __solve(self):

        
        status = self.root.model.optimize()

        if status == OptimizationStatus.INFEASIBLE: # poda por inviabilidade
          
            return None

        node_objective_value = self.root.model.objective_value


        if node_objective_value < BranchAndBound.primal_limit: # poda por limitante
           
            return None
        
        if is_integer_set(self.root.variables): # poda por integralidade
            self.root.is_integer = True
            
            if node_objective_value > BranchAndBound.primal_limit:
                BranchAndBound.primal_limit = node_objective_value
                BranchAndBound.best_node = self.root
                return self.root
         

        else:
            for i in range(len(self.root.variables)):
                if is_integer_variable(self.root.variables[i]) == False:
                    left_model = self.root.model.copy()
                    right_model = self.root.model.copy()

                    left_model += self.root.variables[i] <= 0
                    right_model += self.root.variables[i] >= 1

                    left_model.verbose = 0
                    right_model.verbose = 0

                    self.root.left = BranchAndBound(Node(left_model, left_model.vars, self.root.level + 1), 
                                                    BranchAndBound.__bb_key).__solve()
                    self.root.right = BranchAndBound(Node(right_model, right_model.vars, self.root.level + 1), 
                                                     BranchAndBound.__bb_key).__solve()

                    return self.root