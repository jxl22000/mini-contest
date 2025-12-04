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
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BorderReflexAgent', second = 'BorderReflexAgent'):
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

from baselineTeam import ReflexCaptureAgent

class BorderReflexAgent(ReflexCaptureAgent):
  """
  Offense and Defense border agent: prioritizes easy food and easy defense near the borders.
  """

  def registerInitialState(self, gameState):
    super().registerInitialState(gameState)

    width = gameState.data.layout.width
    if self.red:
      self.borderX = width // 2 - 1
    else:
      self.borderX = width // 2

    self.border = []

    for y in range(gameState.data.layout.height):
      if not gameState.hasWall(self.borderX, y):
        self.border.append((self.borderX, y))
    
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      pos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(pos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  def isWinning(self, gameState):
    return self.getScore(gameState) > 0
  
  def getDefensiveAction(self, gameState):
    return None
  
  def getOffensiveAction(self, gameState):
    return None
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    pos = gameState.getAgentPosition(self.index)

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if gameState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [action.getAgentState(i) for i in self.getOpponents(action)]
    invaders = [a for a in enemies if a.isPacman and a.getAgentPosition(self.index) is not None]
    ghosts = [a for a in enemies if not a.isPacman and a.getAgentPosition(self.index) is not None]

    if len(invaders) > 0:
      features['defense'] = 1
      dists = [self.getMazeDistance(pos, a.getAgentPosition(self.index)) for a in invaders]
      features['invaderDistance'] = min(dists)
    else:
      features['defense'] = 1

    # If there are no invaders, go to the border
    if features['numInvaders'] == 0:
      borderDists = [self.getMazeDistance(pos, borderPos) for borderPos in self.border]
      features['borderDistance'] = min(borderDists)


    foods = self.getFood(action).asList()
    if len(foods) > 0:
        closestFood = min([self.getMazeDistance(pos, f) for f in foods])
    else:
        closestFood = 0

    features["foodDistance"] = closestFood


    borderDist = min([self.getMazeDistance(pos, b) for b in self.border])
    features["borderDistance"] = borderDist



    if len(ghosts) > 0:
      ghostpos = [g.getAgentPosition(self.index) for g in ghosts]
      dists = [self.getMazeDistance(pos, p) for p in ghostpos]
      features["ghostDistance"] = min(dists)

    return features
  
  def getWeights(self, gameState, action):
    return {
      "modeDefense": 100,
      "invaderDistance": -5,
      "foodDistance": -3,
      "borderDistance": -2,
      "ghostDistance": 10
    }

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    return actions[values.index(max(values))]


    # actions = gameState.getLegalActions(self.index)

    # # You can profile your evaluation time by uncommenting these lines
    # # start = time.time()
    # values = [self.evaluate(gameState, a) for a in actions]
    # # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    # maxValue = max(values)
    # bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # foodLeft = len(self.getFood(gameState).asList())

    # if foodLeft <= 2:
    #   bestDist = 9999
    #   for action in actions:
    #     successor = self.getSuccessor(gameState, action)
    #     pos2 = successor.getAgentPosition(self.index)
    #     dist = self.getMazeDistance(self.start,pos2)
    #     if dist < bestDist:
    #       bestAction = action
    #       bestDist = dist
    #   return bestAction

    # return random.choice(bestActions)




class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

