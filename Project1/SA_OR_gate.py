import math
import random
import matplotlib.pyplot as plt

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

def acceptance_probability(old_cost, new_cost, temp):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temp)

def temp_schedule(t, T0, method):
    if method == "logarithmic":
        return T0 / math.log(1+t)
    elif method == "linear":
        return max(T0 - t, 0.001)
    elif method == "exponential":
        return T0 * (0.95 ** t)
    else:
        raise ValueError("Unknown temperature schedule method.")

# Initial state, temp, and parameters
initial_state = (0,0,0)
initial_temp = 100.0
current_state = initial_state 
temp = initial_temp 
cooling_rate = 0.95
min_temp = 1e-3
iter = 1
max_iter = 1000
schedule_method = "logarithmic" #choose between logarithmic, exponential, and linear.
history = [] # To record iterations, temo, current_cost, candidate_cost, and delta_E)


while temp > min_temp and iter <= max_iter:
    new_state = neighbor(current_state)
    old_cost = cost_function(current_state)
    new_cost = cost_function(new_state)
    delta_E = old_cost - new_cost

    if acceptance_probability(old_cost, new_cost, temp) > random.random():
        current_state = new_state
        current_cost = new_cost
    else:
        current_cost = old_cost
    #Record history
    history.append((iter, temp, old_cost, new_cost, delta_E))

    temp = temp_schedule(iter, initial_temp, method=schedule_method)
    iter += 1

#Final solution in current_state
print("Final configuration: ", current_state)
print("Final cost: ", cost_function(current_state)) 

#Compute outputs for each input pair
def or_gate_output(state):
    Wx, Wy, Wb = state
    outputs = {}
    for (x,y) in [(0,0), (0,1),(1,0), (1,1)]:
        net_input = Wx * x + Wy * y + Wb
        output = 1/ (1+ math.exp(-net_input))
        outputs[(x,y)] = output
    return outputs

outputs = or_gate_output(current_state)
print("OR gate outputs: ")
for key, value in outputs.items():
    print(f"Input {key}: Output = {value:.5f}")

#PLot history
iters = [h[0] for h in history]
temps = [h[1] for h in history]
best_costs = [h[2] for h in history]

#Data visualizatioins
plt.figure(figsize=(14,4))

plt.subplot(1,2,1)
plt.plot(iters, temps, label="Temperature", color="blue")
plt.xlabel("Iterations")
plt.ylabel("temperature")
plt.title("Temperature vs Iteration")
plt.legend

plt.subplot(1,2,2)
plt.plot(iters, best_costs, label="best Cost", color = "red")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Best Cost vs Iteration")
plt.legend()

plt.tight_layout()
plt.show()