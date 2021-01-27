import logging
import random
import gym

logger = logging.getLogger(__name__)

class GridWorld_DnS(gym.Env):
    metadata = {'render.modes':['human','rgb_array'],'video.frames_per_second':2}
    
    def __init__(self):
        #----------------------Initial states and directions--------------------#
        # self.states = [[0,0], [0,1], [0,2], [0,3], [0,4], [1,0], [1,1], [1,2], [1,3], [1,4], [2,0], [2,1], [2,2], [2,3], [2,4], [3,1], [3,2], [3,3], [3,4], [4,1], [4,2], [4,3], [4,4]]
        self.Terminal = 24      # 5*5 space represented by 1*25 matrix: 0 ~ 24
        self.state = 0
        self.StaticObs1 = 10
        self.StaticObs2 = 19
        self.DynamicObs1 = 21
        self.DynamicObs2 = 14
        self.actions = [0, 1, 2, 3]     # Direction: Up_0, Down_1, Left_2, Right_3
        self.Obs1Dir = 0
        self.Obs2Dir = 2
        self.gamma = 0.8
        self.viewer = None
        #-----------------------------------------------------------------------#


        #-----------------------Coordinates for ploting-------------------------#
        self.x = [150,250,350,450,550] * 5
        self.y = [550] * 5 + [450] * 5 + [350] * 5 + [250] * 5 + [150] * 5
        #-----------------------------------------------------------------------#


        #----------------------Episodes stopping criterion----------------------#
        self.TerminateStates = dict()      # When crash into obstacles or arrive terminal
        self.TerminateStates[self.StaticObs1] = 1
        self.TerminateStates[self.StaticObs2] = 1
        self.TerminateStates[self.DynamicObs1] = 1
        self.TerminateStates[self.DynamicObs2] = 1
        self.TerminateStates[self.Terminal] = 1
        #-----------------------------------------------------------------------#


        #-------------------------------Rewards---------------------------------#
        self.Rewards = dict()   # Keys are combined with 'states' and '_actions'
        self.Rewards[str(self.DynamicObs1 - 5) + '_1'] = -200.0
        self.Rewards[str(self.DynamicObs1 - 1) + '_3'] = -200.0
        self.Rewards[str(self.DynamicObs1 + 1) + '_2'] = -200.0
        self.Rewards[str(self.DynamicObs2 - 5) + '_1'] = -200.0
        self.Rewards[str(self.DynamicObs2 - 1) + '_3'] = -200.0
        self.Rewards[str(self.DynamicObs2 + 5) + '_0'] = -200.0
        self.Rewards[str(self.StaticObs1 - 5) + '_1'] = -200.0
        self.Rewards[str(self.StaticObs1 + 1) + '_2'] = -200.0
        self.Rewards[str(self.StaticObs1 + 5) + '_0'] = -200.0
        self.Rewards[str(self.StaticObs2 - 5) + '_1'] = -200.0
        self.Rewards[str(self.StaticObs2 - 1) + '_3'] = -200.0
        self.Rewards[str(self.StaticObs2 + 5) + '_0'] = -200.0
        self.Rewards[str(self.Terminal - 5) + '_1'] = 100.0
        self.Rewards[str(self.Terminal - 1) + '_3'] = 100.0
        #-----------------------------------------------------------------------#


        #-----------------------Transformation Dictionary-----------------------#
        self.size = 5
        self.T = dict()
        for i in range(self.size, self.size * self.size):
            self.T[str(i) + '_0'] = i - 5   # States that can go up

        for i in range(self.size * (self.size - 1)):
            self.T[str(i) + '_1'] = i + 5   # States that can go down

        for i in range(1, self.size * self.size):
            if i % self.size == 0:
                continue
            self.T[str(i) + '_2'] = i - 1   # States that can go left  
        
        for i in range(self.size * self.size):
            if (i + 1) % self.size == 0:
                continue
            self.T[str(i) + '_3'] = i + 1   # States that can go right
        #-----------------------------------------------------------------------#


    #----------------------------Step Function------------------------------#
    def step(self, action):

        self.temp = dict()
        self.temp[self.DynamicObs1] = 1
        self.temp[self.DynamicObs2] = 1

        # Update terminate states
        self.TerminateStates.pop(self.DynamicObs1)
        self.TerminateStates.pop(self.DynamicObs2)

        if self.Obs1Dir == 0:
            if self.DynamicObs1 == 1:
                self.Obs1Dir = 1
                self.DynamicObs1 += 5
            else:
                self.DynamicObs1 -= 5
        else:
            if self.DynamicObs1 == 21:
                self.DynamicObs1 -= 5
                self.Obs1Dir = 0
            else:
                self.DynamicObs1 += 5
        
        if self.Obs2Dir == 2:
            if self.DynamicObs2 == 12:
                self.DynamicObs2 += 1
                self.Obs2Dir = 3
            else:
                self.DynamicObs2 -= 1
        else:
            if self.DynamicObs2 == 14:
                self.Obs2Dir = 2
                self.DynamicObs2 -= 1
            else:
                self.DynamicObs2 += 1
        
        self.TerminateStates[self.DynamicObs1] = 1
        self.TerminateStates[self.DynamicObs2] = 1

        # Update rewards dictionary
        self.Rewards = dict()
        if self.DynamicObs1 == 21:
            self.Rewards[str(self.DynamicObs1 - 5) + '_1'] = -200.0
            self.Rewards[str(self.DynamicObs1 - 1) + '_3'] = -200.0
            self.Rewards[str(self.DynamicObs1 + 1) + '_2'] = -200.0
        elif self.DynamicObs1 == 1:
            self.Rewards[str(self.DynamicObs1 + 5) + '_0'] = -200.0
            self.Rewards[str(self.DynamicObs1 - 1) + '_3'] = -200.0
            self.Rewards[str(self.DynamicObs1 + 1) + '_2'] = -200.0
        else:
            self.Rewards[str(self.DynamicObs1 - 5) + '_1'] = -200.0
            self.Rewards[str(self.DynamicObs1 - 1) + '_3'] = -200.0
            self.Rewards[str(self.DynamicObs1 + 1) + '_2'] = -200.0
            self.Rewards[str(self.DynamicObs1 + 5) + '_0'] = -200.0
    
        if self.DynamicObs2 == 14:
            self.Rewards[str(self.DynamicObs2 - 5) + '_1'] = -200.0
            self.Rewards[str(self.DynamicObs2 - 1) + '_3'] = -200.0
            self.Rewards[str(self.DynamicObs2 + 5) + '_0'] = -200.0
        else:
            self.Rewards[str(self.DynamicObs2 - 5) + '_1'] = -200.0
            self.Rewards[str(self.DynamicObs2 - 1) + '_3'] = -200.0
            self.Rewards[str(self.DynamicObs2 + 1) + '_2'] = -200.0
            self.Rewards[str(self.DynamicObs2 + 5) + '_0'] = -200.0
    
        self.Rewards[str(self.StaticObs1 - 5) + '_1'] = -200.0
        self.Rewards[str(self.StaticObs1 + 1) + '_2'] = -200.0
        self.Rewards[str(self.StaticObs1 + 5) + '_0'] = -200.0
        self.Rewards[str(self.StaticObs2 - 5) + '_1'] = -200.0
        self.Rewards[str(self.StaticObs2 - 1) + '_3'] = -200.0
        self.Rewards[str(self.StaticObs2 + 5) + '_0'] = -200.0
        self.Rewards[str(self.Terminal - 5) + '_1'] = 100.0
        self.Rewards[str(self.Terminal - 1) + '_3'] = 100.0

        state = self.state
        key = "%d_%d"%(state,action)

        # Dectect whether this action will lead to crashing into the wall
        if key in self.T:
            next_state = self.T[key]
        else:
            next_state = state
            r = -200.0
            is_Done = True
            return next_state, r, is_Done, {}
        
        # Dectect whether this action will lead to crashing into the obstacles
        self.state = next_state
        is_Done = False
        if next_state in self.TerminateStates or (next_state in self.temp and state in self.TerminateStates):
            is_Done = True
        
        if key not in self.Rewards:
            if (self.Terminal - next_state) < (self.Terminal - state):
                r = 20.0
            else:
                r = -50.0
        else:
            r = self.Rewards[key]
        
        return next_state, r, is_Done, {}
    #-----------------------------------------------------------------------#


    #--------------------------Reset Function-------------------------------#
    def reset(self):

        # Reset states and directions
        self.Terminal = 24
        self.state = 0
        self.StaticObs1 = 10
        self.StaticObs2 = 19
        self.DynamicObs1 = 21
        self.DynamicObs2 = 14
        self.actions = [0, 1, 2, 3]
        self.Obs1Dir = 0
        self.Obs2Dir = 2
        self.gamma = 0.8
        # self.viewer = None

        # Reset episodes stopping criterion
        self.TerminateStates = dict()
        self.TerminateStates[self.StaticObs1] = 1
        self.TerminateStates[self.StaticObs2] = 1
        self.TerminateStates[self.DynamicObs1] = 1
        self.TerminateStates[self.DynamicObs2] = 1
        self.TerminateStates[self.Terminal] = 1

        # Reset rewards dictionary
        self.Rewards = dict()
        self.Rewards[str(self.DynamicObs1 - 5) + '_1'] = -200.0
        self.Rewards[str(self.DynamicObs1 - 1) + '_3'] = -200.0
        self.Rewards[str(self.DynamicObs1 + 1) + '_2'] = -200.0
        self.Rewards[str(self.DynamicObs2 - 5) + '_1'] = -200.0
        self.Rewards[str(self.DynamicObs2 - 1) + '_3'] = -200.0
        self.Rewards[str(self.DynamicObs2 + 5) + '_0'] = -200.0
        self.Rewards[str(self.StaticObs1 - 5) + '_1'] = -200.0
        self.Rewards[str(self.StaticObs1 + 1) + '_2'] = -200.0
        self.Rewards[str(self.StaticObs1 + 5) + '_0'] = -200.0
        self.Rewards[str(self.StaticObs2 - 5) + '_1'] = -200.0
        self.Rewards[str(self.StaticObs2 - 1) + '_3'] = -200.0
        self.Rewards[str(self.StaticObs2 + 5) + '_0'] = -200.0
        self.Rewards[str(self.Terminal - 5) + '_1'] = 100.0
        self.Rewards[str(self.Terminal - 1) + '_3'] = 100.0

        return self
    #-----------------------------------------------------------------------#


    #-----------------------------Rendering---------------------------------#
    def render(self, mode = 'human'):
        from gym.envs.classic_control import rendering
        screen_width = 700
        screen_height = 700

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width,screen_height)

            # Plot the GridWorld
            self.line1 = rendering.Line((100,100),(600,100))
            self.line2 = rendering.Line((100, 200), (600, 200))
            self.line3 = rendering.Line((100, 300), (600, 300))
            self.line4 = rendering.Line((100, 400), (600, 400))
            self.line5 = rendering.Line((100, 500), (600, 500))
            self.line6 = rendering.Line((100, 600), (600, 600))

            self.line7 = rendering.Line((100, 100), (100, 600))
            self.line8 = rendering.Line((200, 100), (200, 600))
            self.line9 = rendering.Line((300, 100), (300, 600))
            self.line10 = rendering.Line((400, 100), (400, 600))
            self.line11 = rendering.Line((500, 100), (500, 600))
            self.line12 = rendering.Line((600, 100), (600, 600))


            # Plot dynamic obstacle_1
            self.obs1 = rendering.make_circle(40)
            self.obs1trans = rendering.Transform()    # translation=(250, 150)
            self.obs1.add_attr(self.obs1trans)
            self.obs1.set_color(1, 0, 0)

            # Plot dynamic obstacle_2
            self.obs2 = rendering.make_circle(40)
            self.obs2trans = rendering.Transform()
            self.obs2.add_attr(self.obs2trans)
            self.obs2.set_color(1, 0, 0)

            # Plot static obstacle_1
            self.obstacle_1 = rendering.make_circle(40)
            self.obstacle1trans = rendering.Transform()
            self.obstacle_1.add_attr(self.obstacle1trans)
            self.obstacle_1.set_color(0, 0, 0)

            # Plot static obstacle_2
            self.obstacle_2 = rendering.make_circle(40)
            self.obstacle2trans = rendering.Transform()
            self.obstacle_2.add_attr(self.obstacle2trans)
            self.obstacle_2.set_color(0, 0, 0)

            # Plot Terminal
            self.terminal = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(550, 150))
            self.terminal.add_attr(self.circletrans)
            self.terminal.set_color(0, 0, 1)

            # Plot robot
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0, 1, 0)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.obs1)
            self.viewer.add_geom(self.obs2)
            self.viewer.add_geom(self.obstacle_1)
            self.viewer.add_geom(self.obstacle_2)
            self.viewer.add_geom(self.terminal)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None

        self.robotrans.set_translation(self.x[self.state], self.y[self.state])
        self.obs1trans.set_translation(self.x[self.DynamicObs1], self.y[self.DynamicObs1])
        self.obs2trans.set_translation(self.x[self.DynamicObs2], self.y[self.DynamicObs2])
        self.obstacle1trans.set_translation(self.x[self.StaticObs1], self.y[self.StaticObs1])
        self.obstacle2trans.set_translation(self.x[self.StaticObs2], self.y[self.StaticObs2])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        #-----------------------------------------------------------------------#

