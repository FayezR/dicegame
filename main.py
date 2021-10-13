from examplealgorithms import DiceGameAgent
from numpy import absolute


class MyAgent(DiceGameAgent):
    def __init__(self, game):
        super().__init__(game)
        self.gamma = 1
        self.theta = 0.99

        # As part of pre-processing, a dictionary of states (key) and the expected reward and optimal policy [
        # expected reward, optimal policy] (key) is created. This dictionary is used to store the optimal policy for
        # any given state
        self.policies = self.value_iteration()

    def value_iteration(self):

        # Initialises dictionary of states (key) and the expected reward and optimal policy
        # [expected reward, optimal policy] (value)
        policies = {}
        for state in self.game.states:
            policies.update({state: [0, None]})

        while True:
            delta = 0

            for state in self.game.states:
                actions_dict = self.find_next_step(policies, state)

                v = actions_dict.values()  # Creates a list of expected reward values for a given action
                highest_action_reward = max(v)  # Finds the biggest reward from the list of expected reward values

                # Stores the absolute difference between new max reward value and previous max reward value for a state
                x = absolute(highest_action_reward - policies[state][0])
                delta = max(delta, x)  # delta is updated with the maximum value of delta or x

                y = max(actions_dict, key=actions_dict.get)
                policies[state] = [highest_action_reward, y]

            if self.theta > delta:
                break

        return policies

    def find_next_step(self, policies, state):

        #  Creates a dictionary to store actions (key) and their expected reward (probability * reward) (value)
        actions_dict = {}
        for n in self.game.actions:
            actions_dict.update({n: 0})

        actions = actions_dict.keys()  # generates a list of actions from the dictionary

        for action in actions:
            next_states, game_over, current_reward, probabilities = self.game.get_next_states(action, state)

            for x in range(len(next_states)):
                probability = probabilities[x]
                next_state = next_states[x]

                if next_state is None:  # if no next_state, probability is multiplied by reward of the current state
                    x = probability * current_reward
                    actions_dict[action] = actions_dict[action] + x

                else:  # i.e. if there is a next_state, the Bellman Update is calculated
                    bellman = probability * (current_reward + self.gamma * policies[next_state][0])
                    actions_dict[action] = actions_dict[action] + bellman

        return actions_dict

    def play(self, state):
        """
        Function returns an optimal action for a given state. 
        The function looks up the dictionary value for a given state returns the optimal policy.
        The dictionary is set up to store a list of values for a given state.
        :policies[state][0]   returns the expected reward for an action
        :policies[state][1]   returns the optimal policy for the state
        """

        return self.policies[state][1]
