from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np

from examplealgorithms import DiceGameAgent


class MyAgent(DiceGameAgent):
    def __init__(self, game):
        super().__init__(game)
        self.gamma = 1
        self.theta = 0.1
        self.value = self.value_iteration()

    def find_next_step(self, value, state):

        #  Creates a dictionary to store actions (key) and probability*reward (value)
        actions_dict = {}
        for single_action in self.game.actions:
            actions_dict.update({single_action : 0})

        # Creates a list of actions from the dictionaru
        actions = actions_dict.keys()

        # For each single_action from the list of actions
        for single_action in actions:
            next_states, game_over, current_reward, probabilities = self.game.get_next_states(single_action, state)

            # For each next_state and associated probability of that action & given state
            for next_state, probability in zip(next_states, probabilities):
                if  next_state == None :  # if there is no next_state then multiply probability by reward of the current state
                    x = probability * current_reward
                    actions_dict[single_action] += x

                else:  # Calculate Bellman Update if there is a next state
                    bellman = probability * (current_reward + self.gamma * value[next_state][0])
                    actions_dict[single_action] += bellman

        return actions_dict

    def value_iteration(self):

        # Initialise dictionary with each state as keys and values [expected reward, optimal policy]
        value = {}
        for state in self.game.states:
            value.update({state: [0, None]})
        # Loop until delta < self.theta
        while True:
            delta = 0

            # For each state
            for state in self.game.states:
                # Get the list of reward values pertaining to each action given that state
                actions = self.find_next_step(value, state)

                # Take the highest expected reward value
                highest_action_reward = max(actions.values())

                # Update delta with either delta or the absolute difference between the
                # new maximum value for that state and the previous maximum value for that state
                delta = max(delta, np.abs(highest_action_reward - value[state][0]))

                # Update the maximum value and optimal policy for that state
                value[state] = [highest_action_reward, max(actions, key=actions.get)]

            # If the change is less than self.theta, end the loop
            if delta < self.theta:
                break
        return value

    def play(self, state):
        """
        For the given state, obtains the optimal policy

        param state : the current state of the game

        returns self.V[state][1] : the optimal action associated with that state
        """

        return self.value[state][1]

