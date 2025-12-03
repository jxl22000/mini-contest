# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BalancedReflexAgent', second = 'BalancedReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.home_boundary = collect_home_boundary(self, gameState)
        CaptureAgent.registerInitialState(self, gameState)

    def get_other_team_in_range(self, agentIndex, state, max_dist=10000):
        """
        Get the agents in other team (different from the team of agentIndex)
        and keep only those that are within max_dist
        """
        myTeam = self.getTeam(state)
        enemyTeam = self.getOpponents(state)
        if agentIndex in myTeam:
            otherTeam = enemyTeam
        elif agentIndex in enemyTeam:
            otherTeam = myTeam
        else:
            raise Exception('This should not happen.')

        pos = state.getAgentState(agentIndex).getPosition()
        otherTeam = [
            idx for idx in otherTeam
            if self.getMazeDistance(
                pos, state.getAgentState(idx).getPosition()) <= max_dist]
        return otherTeam

    def getMax(self, state, agentIndex, depth, bestMax, bestMin):
        if depth <= 0:
            return self.evaluate(state)
        possibleActions = state.getLegalActions(agentIndex)
        if Directions.STOP in possibleActions:  # Our agent never stops
            possibleActions.remove(Directions.STOP)

        bestScore = -1e12
        for action in possibleActions:
            successor = state.generateSuccessor(agentIndex, action)
            otherTeam = self.get_other_team_in_range(agentIndex, state)
            if len(otherTeam) > 0:
                score = min([
                    self.getMin(
                        successor, nextAgent, depth-1, bestMax, bestMin)
                    for nextAgent in otherTeam])
            else:
                # assume the other team won't move at all
                score = self.evaluate(successor)
            bestScore = max(bestScore, score)
            if bestScore > bestMin:  # The min agent will never allow this
                return bestScore
            bestMax = max(bestMax, bestScore)
        return bestScore

    def getMin(self, state, agentIndex, depth, bestMax, bestMin):
        if depth <= 0:
            return self.evaluate(state)
        possibleActions = state.getLegalActions(agentIndex)

        bestScore = 1e12
        for action in possibleActions:
            successor = state.generateSuccessor(agentIndex, action)
            otherTeam = self.get_other_team_in_range(agentIndex, state)
            if len(otherTeam) > 0:
                score = max([
                    self.getMax(
                        successor, nextAgent, depth-1, bestMax, bestMin)
                    for nextAgent in self.getTeam(state)])
            else:
                # assume the other team won't move at all
                score = self.evaluate(successor)
            bestScore = min(bestScore, score)
            if bestScore < bestMax:  # The max agent will never allow this
                return bestScore
            bestMin = min(bestMin, bestScore)
        return bestScore

    def evaluate_minimax(self, state, depth=2):
        """
        Evaluate a state using minimax (alpha-beta pruning)
        state is the successor by YOUR action (so now it is the enemy's turn)
        """
        bestMax, bestMin = -1e12, 1e12
        return self.getMin(state, self.index, depth, bestMax, bestMin)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest values.
        """
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [
            # self.evaluate(self.getSuccessor(gameState, a)) for a in actions
            self.evaluate_minimax(self.getSuccessor(gameState, a)) for a in actions
        ]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        # Take stochastic actions when the values are close.
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        # bestActions = [np.random.choice(actions, p=softmax(values))]
        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState):
        """
        Computes a linear combination of features and feature weights
        """
        features, weights = self.getFeaturesAndWeights(gameState)
        return features * weights

    def getFeaturesAndWeights(self, gameState):
        """
        Returns a counter of features for the state

        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        raise NotImplementedError()


class BalancedReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeaturesAndWeights(self, state):
        weights = {}

        features = util.Counter()
        foodList = self.getFood(state).asList()
        capsuleList = self.getCapsules(state)

        myState = state.getAgentState(self.index)
        myPos = myState.getPosition()
        dist_to_home = mazeDistanceToHome(self, myPos, state)

        enemies = [
            state.getAgentState(i) for i in self.getOpponents(state)]
        enemy_active_ghosts = [  # enemy ghosts that are not scared
            a for a in enemies if ((not a.isPacman) and a.scaredTimer < 5)]
        enemy_pacmans = [a for a in enemies if a.isPacman]

        #######################################################################
        # Features for winning
        #######################################################################

        # 'scores' (higher is better):
        #   This is what we ultimately care about. Unfortunately, the search
        #   algorithm cannot directly optimize it due to the limited horizon
        features['scores'] = self.getScore(state)
        weights['scores'] = 5000.

        # 'backToStart' (lower is better):
        #   Back to the start (usually because you're eaten) is really bad
        features['backToStart'] = float(myPos == self.start)
        weights['backToStart'] = -10000.

        #######################################################################
        # Features for collecting food and power capsules
        #######################################################################

        # 'remainingFoodOrCapsule' (lower is better):
        #   number of remaining food or power capsules
        # expected behavior: to collect food or power capsules
        features['remainingFoodOrCapsule'] = len(foodList)+len(capsuleList)
        weights['remainingFoodOrCapsule'] = -1000

        # 'dToClosetFoodOrCapsule' (lower is better):
        #   the (weighted) distance to the closest food or power capsules
        #   this distance is reweighed w.r.t. ghosts
        # expected behavior: to get close to food or power capsules
        if len(foodList) > 0 or len(capsuleList) > 0:
            dist = [
                weighted_food_distance(self, myPos, food, enemy_active_ghosts)
                for food in (foodList + capsuleList)]
            minDistance = min(dist)
            features['dToClosetFoodOrCapsule'] = minDistance
        else:
            features['dToClosetFoodOrCapsule'] = 0.
        weights['dToClosetFoodOrCapsule'] = -10

        # 'carriedFoodDistanceToHome' (lower is better):
        #   the distance to one's own territory, weighted by number of food
        #   carried by the agent
        # expected behavior: to deposit food to home territory
        features['carriedFoodDistanceToHome'] = \
            myState.numCarrying * dist_to_home
        weights['carriedFoodDistanceToHome'] = -16

        # 'nrdToGhost' (higher is better):
        #   the negative reciprocal distance (nrd) to enemy Ghosts (-\infty, 0)
        #   if enemy ghosts are far or nonexistance, nrd is close to zero
        #   if enemy ghosts are close, nrd is close to -\infty
        # expected behavior: to excape from ghosts
        dists = [self.getMazeDistance(myPos, a.getPosition())
                 for a in enemy_active_ghosts]
        minDistance = min(dists + [1e8])
        features['nrdToGhost'] = -1. / max(minDistance - 2, 1e-3)
        weights['nrdToGhost'] = 150

        #######################################################################
        # Features for dealing with invaders
        #######################################################################

        # 'numInvaders' (lower is better):
        #   the number of enemy pacmans on our territory
        # expected behavior: to kill invaders
        features['numInvaders'] = len(enemy_pacmans)
        weights['numInvaders'] = -10000

        # 'dInvader' (lower is better):
        #   the (weighted) distance to the closest enemy invader
        #   (enemies carrying food has higher threat and are targeted first)
        #   if no enemy invader, the distance is 0.
        #   this distance is weighted by the (global) threat weight
        #   there is also a distance to keep when the agent is scared
        # expected behavior: to get close to invaders (keep away if scared)
        keep_dist = min(myState.scaredTimer * 2, 4)
        dists = [
            abs(self.getMazeDistance(myPos, a.getPosition()) - keep_dist)
            + (100 - 4. * a.numCarrying)
            for a in enemy_pacmans]
        # threat decides how urgent it is to deal with invaders.
        # Each invader poses a threat of (1 + #their-carried-food)
        threat = sum([(1+a.numCarrying) for a in enemy_pacmans])
        features['dInvader'] = min(dists + [10000.]) * threat
        weights['dInvader'] = -7

        #######################################################################
        # Features for other desidered behavior
        #######################################################################

        # 'dToCenter' (lower is better):
        #   the distance to the central column of the maze
        #   it may be better to get close to the center if nothing else to do
        #   (e.g. to better defend against invaders)
        # expected behavior: to get closer to center
        features['dToCenter'] = dist_to_home
        weights['dToCenter'] = -1.

        return features, weights


###############################################################################
# Utility functions
###############################################################################

def weighted_food_distance(agent, agent_pos, food_pos, enemy_ghosts):
    food_dist = agent.getMazeDistance(agent_pos, food_pos)

    dists = [agent.getMazeDistance(food_pos, a.getPosition())
             for a in enemy_ghosts]
    minDistance = min(dists + [10000])
    rDistanceToGhost = 1 / max(minDistance - 4, 0.1)

    return food_dist * (1 + rDistanceToGhost)


def softmax(values):
    logits = np.exp(values - np.max(values))
    return logits / np.sum(logits)


def collect_home_boundary(agent, gameState):
    halfway = gameState.data.layout.width // 2
    height = gameState.data.layout.height
    boundary_red = [
        (halfway - 1, y) for y in range(height)
        if not gameState.data.layout.walls[halfway - 1][y]]
    boundary_blu = [
        (halfway, y) for y in range(height)
        if not gameState.data.layout.walls[halfway][y]]
    return {True: boundary_red, False: boundary_blu}


def mazeDistanceToHome(agent, pos, gameState):
    dist = min([agent.getMazeDistance(pos, pos_boundary)
                for pos_boundary in agent.home_boundary[agent.red]])
    return dist
