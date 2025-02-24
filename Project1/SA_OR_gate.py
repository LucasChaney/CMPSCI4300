import math
import random

def cost_function(state):
    #Define the cost based on how well the state simulate the OR gate
    # Compare outputs against the OR truth table
    Wx, Wy, Wb = state
    truth_table = [((0,0),0),
                    ((0,1),1),
                    ((1,0),1),
                    ((1,1),1)]
    cost = 0.0
    for (x,y), target in truth_table:
        net_input = Wx * x + Wy * y + Wb
        #sigmoid activation function
        output = 1 / (1 + math.exp(-net_input))
        cost += (target - output) ** 2
    return cost

def neighbor(state, step_size = 0.1):
    #Return a slightly modified version of the state
    Wx, Wy, Wb = state
    # Generate random changes for eachweight in range of -step_size to step_size
    new_Wx = Wx + random.uniform(-step_size, step_size)
    new_Wy = Wy + random.uniform(-step_size, step_size)
    new_Wb = Wb + random.uniform(-step_size, step_size)

    new_state = (new_Wx, new_Wy, new_Wb)
    return new_state

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)
    
# Initial state, temperature, and parameters
initial_state = (0,0,0)
initial_temperature = 100.0
current_state = initial_state 
temperature = initial_temperature 
cooling_rate = 0.95
min_temperature = 1e-3

while temperature > min_temperature:
    new_state = neighbor(current_state)
    old_cost = cost_function(current_state)
    new_cost = cost_function(new_state)

    if acceptance_probability(old_cost, new_cost, temperature) > random.random():
        current_state = new_state

    temperature *= cooling_rate #update temperature

#Final solution in current_state
print("Final configuration: ", current_state)
