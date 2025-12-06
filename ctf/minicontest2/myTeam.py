# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
import numpy as np

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BaseAgent(CaptureAgent):
  def __init__(self, index, timeForComputing=0.1):
    super().__init__(index, timeForComputing)
    self.start = None
    self.foodCarrying = 0
    self.lastAction = None
    self.teamIndices = []
    self.opponentIndices = []

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.teamIndices = self.getTeam(gameState)
    self.opponentIndices = self.getOpponents(gameState)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
        return successor.generateSuccessor(self.index, action)
    else:
        return successor

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    if myState.isPacman:
        foodList = self.getFood(gameState).asList()
        newFoodList = self.getFood(successor).asList()
        self.foodCarrying += (len(foodList) - len(newFoodList))
    elif not myState.isPacman and self.foodCarrying > 0:
        self.foodCarrying = 0
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
      
  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [choice for choice, value in zip(actions, values) if value == maxValue]
    return random.choice(bestActions)

  def getDangerousGhosts(self, gameState, myPos, threshold=5):
    enemies = [gameState.getAgentState(i) for i in self.opponentIndices]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() is not None]
    dangerous = []
    for ghost in ghosts:
        ghostPos = ghost.getPosition()
        dist = self.getMazeDistance(myPos, ghostPos)
        if dist < threshold and ghost.scaredTimer <= 1:
            dangerous.append((ghost, dist))
    return [ghost[0] for ghost in sorted(dangerous, key=lambda x: x[1])]

  def getClosestItem(self, gameState, items, max_distance=9999):
    myPos = gameState.getAgentPosition(self.index)
    minDist = max_distance
    closest = None
    for item in items:
        dist = self.getMazeDistance(myPos, item)
        if dist < minDist:
            minDist = dist
            closest = item
    return closest, minDist

class OffensiveAgent(BaseAgent):
  def __init__(self, index, timeForComputing=0.1):
    super().__init__(index, timeForComputing)
    self.target = None
    self.lastFoodEaten = None
    self.returning = False
    self.ghostMemory = {}
    self.foodMemory = set()
    self.lastSeenFood = 0
    self.scaredGhosts = []
    self.powerPillTarget = None
    self.lastPowerPillCheck = 0

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    features['carrying'] = myState.numCarrying
    if myState.numCarrying > 0:
        self.returning = True
    elif myPos == self.start:
        self.returning = False

    enemies = [successor.getAgentState(i) for i in self.opponentIndices]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
    powerPellets = self.getCapsules(gameState)
    minGhostDist = float('inf')
    for ghost in ghosts:
        ghostPos = ghost.getPosition()
        dist = self.getMazeDistance(myPos, ghostPos)
        self.ghostMemory[ghostPos] = gameState.getAgentState(self.index).scaredTimer
        if dist < minGhostDist:
            minGhostDist = dist
            closestGhost = ghost

    foodList = self.getFood(gameState).asList()
    if foodList:
        self.lastSeenFood = gameState.data.timeleft
        self.foodMemory.update(foodList)
    
    if self.returning:
        features['distanceToHome'] = self.getMazeDistance(myPos, self.start) / 100.0
        features['onDefense'] = 1 if not myState.isPacman else 0
        return features

    if foodList:
        minFoodDist = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minFoodDist / 1.0

    if minGhostDist < 5 and closestGhost.scaredTimer == 0:
        features['ghostDistance'] = minGhostDist / 5.0

    if powerPellets and (gameState.data.timeleft - self.lastPowerPillCheck) > 20:
        self.powerPillTarget = min(powerPellets, 
                                  key=lambda x: self.getMazeDistance(myPos, x))
        self.lastPowerPillCheck = gameState.data.timeleft

    if self.powerPillTarget and self.powerPillTarget in powerPellets:
        features['powerPillDistance'] = self.getMazeDistance(myPos, self.powerPillTarget) / 20.0
    if action == Directions.STOP:
        features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    weights = util.Counter()
    weights['carrying'] = 100
    weights['distanceToHome'] = -10
    weights['distanceToFood'] = -1
    weights['ghostDistance'] = -100
    weights['powerPillDistance'] = -5
    weights['stop'] = -10
    weights['onDefense'] = 100
    return weights

class DefensiveAgent(BaseAgent):
  def __init__(self, index, timeForComputing=0.1):
    super().__init__(index, timeForComputing)
    self.patrolPoints = []
    self.currentPatrolIndex = 0
    self.invaderPositions = {}
    self.defensiveRadius = 5

  def registerInitialState(self, gameState):
    super().registerInitialState(gameState)
    self.initializePatrolPoints(gameState)

  def initializePatrolPoints(self, gameState):
    """Set up patrol points around our food"""
    food = self.getFoodYouAreDefending(gameState)
    walls = gameState.getWalls()
    width, height = gameState.data.layout.width, gameState.data.layout.height
    foodPositions = [(x, y) for x in range(width) for y in range(height) if food[x][y]]
    patrolPoints = set()
    for fx, fy in foodPositions:
        for dx in range(-self.defensiveRadius, self.defensiveRadius + 1):
            for dy in range(-self.defensiveRadius, self.defensiveRadius + 1):
                x, y = fx + dx, fy + dy
                if (0 <= x < width and 0 <= y < height and 
                    not walls[x][y]):
                    patrolPoints.add((x, y))
    self.patrolPoints = list(patrolPoints)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    enemies = [successor.getAgentState(i) for i in self.opponentIndices]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]
    features['numInvaders'] = len(invaders)
    if invaders:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)
    else:
        if not self.patrolPoints:
            self.initializePatrolPoints(gameState)
        
        target = self.patrolPoints[self.currentPatrolIndex]
        features['patrolDistance'] = self.getMazeDistance(myPos, target)
        # Move to next point if we're close
        if myPos == target or self.getMazeDistance(myPos, target) < 2:
            self.currentPatrolIndex = (self.currentPatrolIndex + 1) % len(self.patrolPoints)
    if action == Directions.STOP:
        features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    weights = util.Counter()
    weights['numInvaders'] = -1000
    weights['invaderDistance'] = -10
    weights['patrolDistance'] = -1
    weights['stop'] = -10
    return weights