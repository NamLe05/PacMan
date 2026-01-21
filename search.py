# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from re import X
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    # Initialize the stack and visited set
    stack = util.Stack();
    visited = set([])
    # Push the start state and empty actions onto the stack
    stack.push((problem.getStartState(), []))  

    # While the stack is not empty pop the current state off the stack
    while not stack.isEmpty():
        # Track current state and actions taken
        state, actions = stack.pop()

        if problem.isGoalState(state):
            return actions

        # Add current state to visited and expand fringe by pushing successors to the stack
        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, actions + [action]))

    return [] # No path found

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Initialize the queue and visited set
    queue = util.Queue();
    visited = set([])
    # Enqueue the start state and empty actions onto the queue
    queue.push((problem.getStartState(), []))  

    # While the queue is not empty dequeue the current state off the queue
    while not queue.isEmpty():
        # Track current state and actions taken
        state, actions = queue.pop()

        if problem.isGoalState(state):
            return actions

        # Add current state to visited and expand fringe by enqueuing successors
        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    queue.push((successor, actions + [action]))

    return [] # No path found   

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Initialize the minimum priority queue and visited set
    pq = util.PriorityQueueWithFunction(lambda x: x[2]);
    visited = set([])
    # Enqueue the start state, empty actions, and zero path cost to the min pq
    pq.push((problem.getStartState(), [], 0)) 

    # While the priority queue is not empty pop the current state off the min pq
    while not pq.isEmpty():
        # Track current state and actions taken
        state, actions, cost = pq.pop()

        if problem.isGoalState(state):
            return actions
            
        # Add current state to visited and expand fringe by enqueuing successors
        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    newCost = cost + stepCost
                    pq.push((successor, actions + [action], newCost))

    return [] # No path found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
 
    # Initialize the minimum priority queue where priority is cost + heuristic 
    # and visited set
    pq = util.PriorityQueueWithFunction(lambda x: x[2] + heuristic(x[0], problem));
    visited = set([])

    # Enqueue the start state, empty actions, and zero path cost to the min pq
    pq.push((problem.getStartState(), [], 0)) 

    # While the priority queue is not empty pop the current state off the min pq
    while not pq.isEmpty():
        # Track current state and actions taken
        state, actions, cost = pq.pop()

        if problem.isGoalState(state):
            return actions
            
        # Add current state to visited and expand fringe by enqueuing successors
        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    newCost = cost + stepCost
                    pq.push((successor, actions + [action], newCost))

    return [] # No path found

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
