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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    stack = util.Stack()
    visited = []
    found_solution = False
    
    # We maintain a parentalMap so we can find which node is parent of which.
    # This will be useful to recover the solution path later
    # Source (idea): https://stackoverflow.com/questions/12864004/tracing-and-returning-a-path-in-depth-first-search
    parentalMap = {}
    
    # Initially, let's expand the start state (it is done separately because
    # this is the only node which we don't have cost or action associated)
    # Check for goal reaching
    start = problem.getStartState()
    if (problem.isGoalState(start)):
        return []
    # Otherwise, push its children into the stack for further expansion later
    visited.append(start)
    children = problem.getSuccessors(start)
    for child in children:
        stack.push(child)
        parentalMap[child] = start
    
    # DFS expansion (LIFO)
    while (not stack.isEmpty()):
        current_state = stack.pop()
        (coord, _, _) = current_state
        
        if (coord in visited):
            # Don't expand already visited nodes 
            continue
        elif (problem.isGoalState(coord)):
            # Solution found, we may break now
            found_solution = True
            visited.append(coord)
            break
        else:
            # Expand a node: get its successors and add them to the stack
            visited.append(coord)
            children = problem.getSuccessors(coord)
            for child in children:
                stack.push(child)
                parentalMap[child] = current_state
    
    if (not found_solution):
        raise "DFS Failed to Find solution!"
    
    # We've found the solution, now we need to backtrace the parentMap
    # so we can recover the path from the start to here
    actions = []
    node = current_state
    while(node != start):
        actions.append(node[1])
        node = parentalMap[node]
        
    # We trasverse in reverse order, so reverse the list to obtain the
    # actual solution path
    actions.reverse()
    
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    visited = []
    found_solution = False
    
    # We maintain a parentalMap so we can find which node is parent of which.
    # This will be useful to recover the solution path later
    # Source (idea): https://stackoverflow.com/questions/12864004/tracing-and-returning-a-path-in-depth-first-search
    parentalMap = {}
    
    # Initially, let's expand the start state (it is done separately because
    # this is the only node which we don't have cost or action associated)
    # Check for goal reaching
    start = problem.getStartState()
    if (problem.isGoalState(start)):
        return []
    # Otherwise, push its children into the stack for further expansion later
    visited.append(start)
    children = problem.getSuccessors(start)
    for child in children:
        queue.push(child)
        parentalMap[child] = start
    
    # BFS expansion (FIFO)
    while (not queue.isEmpty()):
        current_state = queue.pop()
        (coord, _, _) = current_state
        
        if (coord in visited):
            # Don't expand already visited nodes 
            continue
        elif (problem.isGoalState(coord)):
            # Solution found, we may break now
            found_solution = True
            visited.append(coord)
            break
        else:
            # Expand a node: get its successors and add them to the stack
            visited.append(coord)
            children = problem.getSuccessors(coord)
            for child in children:
                queue.push(child)
                parentalMap[child] = current_state
    
    if (not found_solution):
        raise "BFS Failed to Find solution!"
    
    # We've found the solution, now we need to backtrace the parentMap
    # so we can recover the path from the start to here
    actions = []
    node = current_state
    while(node != start):
        actions.append(node[1])
        node = parentalMap[node]
        
    # We trasverse in reverse order, so reverse the list to obtain the
    # actual solution path
    actions.reverse()
    
    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    heap = util.PriorityQueue()
    visited = []
    total_distance = {}
    
    # Here we must use a different approach for recovering the path, because
    # we don't have access to enough heap attributes to allow our backtracking.
    # Now, we  will store the path in the structure, along the node itself
    
    start = problem.getStartState()
    heap.push((start, []), 0)     # Heap stores (node_coords, path_to_get_there)
    total_distance[start] = 0
    
    # UCS expansion
    while (not heap.isEmpty()):
        (state, path) = heap.pop()
        partial_cost = total_distance[state]   # Cost to get to this node
        
        if (state in visited):
            # Don't expand already visited nodes 
            continue
        elif (problem.isGoalState(state)):
            # Solution found, we just return the stored path
            return path
        else:
            # Expand a node, add (or update) its children in heap
            visited.append(state)
            children = problem.getSuccessors(state)
            for (c_coord, c_action, c_cost) in children:
                if (c_coord in total_distance and total_distance[c_coord] <= partial_cost + c_cost):
                    # New path to this child node is not better
                    continue
                elif (c_coord in total_distance):
                    # New path to this child node is better, 
                    # We must update the heap with the new cost
                    heap.update((c_coord, path+[c_action]), partial_cost + c_cost)
                else:
                    # This child node is not in heap, we must place it there
                    heap.push((c_coord, path+[c_action]), partial_cost + c_cost)
                    total_distance[c_coord] = partial_cost + c_cost
    
    # If we got here, we weren't able to find a solution
    raise "UCS Failed to Find solution!"
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def greedySearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest heuristic first."""
    heap = util.PriorityQueue()
    visited = []
    heur_distance = {}
    
    start = problem.getStartState()
    heur_distance[start] = heuristic(start, problem)
    heap.push((start, []), heur_distance[start])     
        # Heap stores (node_coords, path_to_get_there)
        # Priority based on heuristic cost
    
    # Best First Search expansion
    while (not heap.isEmpty()):
        (state, path) = heap.pop()
        
        if (state in visited):
            # Don't expand already visited nodes 
            continue
        elif (problem.isGoalState(state)):
            # Solution found, we just return the stored path
            return path
        else:
            # Expand a node, add (or update) its children in heap
            visited.append(state)
            children = problem.getSuccessors(state)
            for (c_coord, c_action, _) in children:
                c_cost = heuristic(c_coord, problem)
                heur_distance[c_coord] = c_cost
                # Update works as push if node isn't there
                heap.update((c_coord, path+[c_action]), c_cost)
    
    # If we got here, we weren't able to find a solution
    raise "Best-First Search (Greedy) Failed to Find solution!"


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    heap = util.PriorityQueue()
    visited = []
    real_distance = {}
    heur_distance = {}
    
    # We'll combine UCS with GreedySearch
    
    start = problem.getStartState()
    heap.push((start, []), 0)     # Heap stores (node_coords, path_to_get_there)
    real_distance[start] = 0
    heur_distance[start] = heuristic(start, problem)
    heap.push((start, []), real_distance[start] + heur_distance[start])     
    
    while (not heap.isEmpty()):
        (state, path) = heap.pop()
        real_partial = real_distance[state]
        # heur_partial = heur_distance[state]
        
        if (state in visited):
            # Don't expand already visited nodes 
            continue
        elif (problem.isGoalState(state)):
            # Solution found, we just return the stored path
            return path
        else:
            # Expand a node, add (or update) its children in heap
            visited.append(state)
            children = problem.getSuccessors(state)
            for (c_coord, c_action, c_cost) in children: 
                if (c_coord in real_distance and real_distance[c_coord] <= real_partial + c_cost):
                    # New real cost to this child node is not better
                    continue
                elif (c_coord in real_distance):
                    # New real cost to this child node is better, 
                    # We must update the heap with the new cost
                    heap.update((c_coord, path+[c_action]), 
                                real_partial + c_cost + heuristic(c_coord, problem))
                else:
                    # This child node is not in heap, we must place it there
                    heap.push((c_coord, path+[c_action]), 
                              real_partial + c_cost + heuristic(c_coord, problem))
                real_distance[c_coord] = real_partial + c_cost
    
    # If we got here, we weren't able to find a solution
    raise "UCS Failed to Find solution!"


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    p=state[0]
    foodGrid = state[1]
    "*** YOUR CODE HERE ***"
    
    foods=foodGrid.asList()
    foodDistance=0
    stateToFood=0
    #print foods
    for f1 in foods:
        for f2 in foods:
            if (f1,f2) in problem.heuristicInfo:
                if problem.heuristicInfo[(f1,f2)]>foodDistance:
                    foodDistance=problem.heuristicInfo[(f1,f2)]
            elif (f2,f1) in problem.heuristicInfo:
                if problem.heuristicInfo[(f2,f1)]>foodDistance:
                    foodDistance=problem.heuristicInfo[(f2,f1)]
            else:
                #print (f1,f2)
                problem.heuristicInfo[(f1,f2)]=mazeDistance(f1,f2,problem.startingGameState)
                if problem.heuristicInfo[(f1,f2)]>foodDistance:
                    foodDistance=problem.heuristicInfo[(f1,f2)]
    
    for f in foods:
        if (p,f) in problem.heuristicInfo:
            if stateToFood ==0 or stateToFood>problem.heuristicInfo[(p,f)]:
                stateToFood=problem.heuristicInfo[(p,f)]
        elif (f,p) in problem.heuristicInfo:
            if stateToFood ==0 or stateToFood >problem.heuristicInfo[(f,p)]:
                stateToFood=problem.heuristicInfo[(f,p)]
        else:
            problem.heuristicInfo[(p,f)]=mazeDistance(p,f,problem.startingGameState)
            if stateToFood==0 or stateToFood>problem.heuristicInfo[(p,f)]:
                stateToFood=problem.heuristicInfo[(p,f)]
    return foodDistance+stateToFood

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.
    Example usage: mazeDistance( (2,4), (5,6), gameState)
    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    
    from searchAgents import PositionSearchProblem
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(bfs(prob))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
ucs = uniformCostSearch
gs = greedySearch
astar = aStarSearch
