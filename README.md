# Offline path planning
In offline mode, the path planning task starts from a point where the agent has a complete knowledge about:
it’s initial position, and it’s final goal within , obstacles in the given map within the environment.The offline
path planing task simply finds the most optimal path connecting the start position to the goal position.Offline
path planning is generally implemented for static environment or very slowly changing environment.Offline
mode will be computationally expensive for dynamic environment as it has to re-plan from start -to-goal path
again making it inefficient and impractical in dynamic environments.



# Online path planning
Online here means that the robot does not have a prior knowledge of all the obstacles in the environment.The
robot moves and only observes when physically in a neighboring cell to the obstacle.The modification is es-
sentially in visualization of environment which in this case is local or myopic . The obstacle is added to the
map only when the robot reaches the in the neighbouring cell to provide a sense of online replanning while
incrementally building an obstacle map.Hence when the robot plans the optimal path till goal point, obstacles
which come in the way while planning are only seen in the environment map.Online is preferred in dynamically
changing environment as path planning is carried out in parallel while the robot moves towards the goal and
observes the environment including its changes.Hence path is updated as the environment changes.
