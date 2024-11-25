from experiments.experiment import Experiment
from typing import List

def find_experiments() -> List[Experiment]:
    paths = []
    experiments = []
    for path in paths:
       experiments.append(Experiment(filepath=path))

    return experiments

def select_experiment():
    experiments = find_experiments() # locate experiments

    print("Please select an experiment : ") #display found experiments
    
    for experiment in range(len(experiments)):
        print(f"{experiments + 1}. {experiments[experiment].name}")

    try:
       user_input = int(input("Selection : ")) #prompt user 
       if user_input >= len(experiments) or user_input < 0:
           print(f"Out of range. Please select between [ 1-{len(experiments)+1} ]" )
    except ValueError:
        print("Invalid integer... ")

    chosen = experiments
    run()

    return

     

#MAIN PROGRAM STARTS HERE:
age = inputNumber("How old are you?")
def run():



    return