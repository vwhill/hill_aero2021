# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:43:20 2020

@author: vince
"""

import numpy as np

class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
        x and y are Cartesian coordinates of the node
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.x = 0
        self.y = 0
    def __eq__(self, other):
        return self.position == other.position

# This function return the path of the search
def return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze)
    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 0
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1
    return result


def search(maze, cost, start, end):
    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param cost
        :param start:
        :param end:
        :return:
    """

    # Create start and end node with initized values for g, h and f
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration. 
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []  
    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = [] 
    
    # Add the start node
    yet_to_visit_list.append(start_node)
    
    # Adding a stop condition. This is to avoid any infinite loop and stop 
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . search movement is left-right-top-bottom 
    #(4 movements) from every positon

    move  =  [[-1, 0 ], # go up
              [ 0, -1], # go left
              [ 1, 0 ], # go down
              [ 0, 1 ], # go right
              [ 1, 1 ], # diagonals
              [ 1, -1],
              [-1, 1 ],
              [-1, -1]]
                


    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit list and add this node to visited list
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent
            
            For all the children node
                a) if child in visited list then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit list then ignore it
                d) else move the child to yet_to_visit list
    """
    #find maze has got how many rows and columns 
    no_rows, no_columns = np.shape(maze)
    
    # Loop until you find the end
    
    while len(yet_to_visit_list) > 0:
        
        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1    

        
        # Get the current node
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
            # while (item.f < current_node.f).any() is True:
                current_node = item
                current_index = index
                
        # if we hit this point return the path such as it may be no solution or 
        # computation cost is too high
        if outer_iterations > max_iterations:
            print ("giving up on pathfinding too many iterations")
            return return_path(current_node,maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return return_path(current_node, maze)

        # Generate children from all adjacent squares
        children = []

        for new_position in move: 

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or 
                node_position[0] < 0 or 
                node_position[1] > (no_columns -1) or 
                node_position[1] < 0 or 
                node_position[0] < 0 and node_position[1] < 0 or 
                node_position[0] > (no_rows-1) and node_position[1] < 0 or
                node_position[0] < 0 and node_position[1] < (no_columns-1) or
                node_position[0] > (no_rows-1) and node_position[1] > (no_columns-1)):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            
            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + cost
            ## Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - end_node.position[0]) ** 2) + 
                       ((child.position[1] - end_node.position[1]) ** 2)) 

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)


def mazegen(a, b):    
    maze = np.random.random((a, b)) # generate random maze
    for ii in range(0, np.size(maze, axis=0)):
        for jj in range(0, np.size(maze, axis=1)):
            if maze[ii][jj] < 0.1:
                maze[ii][jj] = 1
            else:
                maze[ii][jj] = 0
    
    # maze = np.zeros((a, b))
    # maze[:, np.int(b/3)] = 1
    # maze[np.int(2*a/3):a, np.int(b/3)] = 0
    # maze[:, np.int(2*b/3)] = 1
    # maze[0:np.int(a/3), np.int(2*b/3)] = 0
    # maze[np.int(a/5), :] = 1
    # maze[np.int(a/5), np.int(3*b/4):b] = 0
    # maze[np.int(3*a/4), :] = 1
    # maze[np.int(3*a/4), 0:np.int(b/4)] = 0
    # maze[:, np.int(b/2)] = 1
    # maze[np.int(2*a/3):a, np.int(b/2)] = 0
    
    obs = np.where(maze==1)
    
    # start = [0, 0] # starting position
    # end = [a-1, b-1] # ending position
    # maze[start[0]][start[1]] = 0
    # maze[end[0]][end[1]] = 0
    
    cost = 1 # cost per movement
    return maze, cost, obs

def meshgen(a, b, mapsize):
    x=np.linspace(0,mapsize,a)
    y=np.linspace(0,mapsize,b)
    
    mesh = np.zeros((a, b))
    mesh[:, 0] = y
    mesh[0, :] = x
    
    mesh[a-1, :] = np.linspace(mapsize, np.sqrt(2*mapsize**2), a)
    mesh[:, b-1] = np.linspace(mapsize, np.sqrt(2*mapsize**2), b)
    
    for i in range(1,a):
        mesh[i, :] = np.linspace(mesh[i, 0], mesh[i, b-1], b)
    return mesh, x, y

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

# if __name__ == '__main__':
    
#     a = 100
#     b = 100
#     mapsize = 3000 # square size of map
    
#     [maze, cost, obs] = mazegen(a, b)
#     mesh = meshgen(a, b, mapsize)

#     path = search(maze, cost, start, end)
    # print(path)