'''
Author: Ashlee Tiwari
Email: ashleetiwari2021@u.northwestern.edu
'''
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import matplotlib.cm as cm
import math
from matplotlib.ticker import MultipleLocator

class Pose():
    def __init__(self,x,y,theta=-np.pi/2):
        # start position
        self.x=x
        self.y=y
        self.theta=theta

class Node():
    def __init__(self,x,y,cost):
        self.x=x
        self.y=y
        self.cost=cost
      
    def expand(self): 
        results = [(self.x+1,self.y),(self.x,self.y-1),(self.x-1,self.y),(self.x,self.y+1),(self.x+1,self.y+1),(self.x-1,self.y-1),(self.x-1,self.y+1),(self.x+1,self.y-1)]
        return results

class inverse_kinematic():
    '''  outputs translational and rotational speeds [ν ,ω ] that will allow the robot to achieve a target 2-D position'''
    def __init__(self,v=0,w=0):
        '''
         initializes with robot's starting velocity
        '''
        self.v=v
        self.w=w 
        self.delta_t=0.1
        
    
    def estimate_state(self,v,w,delta_t):
        '''     
            Input
                previous state: [x_t-1, y_t-1, θ_t-1]
                control inputs: [v, w]
            Output
                new state estimate :[x_t, y_t, θ_t]
            
            θ_t  =  θ_t-1 + w * delta_t
            x_t  =  x_t-1 + v * delta_t* cos(θ_t-1 + w * delta_t)
            y_t  =  y_t-1 + v * delta_t* sin(θ_t-1 + w * delta_t)
        '''
        self.theta=self.theta+w*(delta_t)
        self.x=self.x+v*np.cos(self.theta)*delta_t
        self.y=self.y+v*np.sin(self.theta)*delta_t         
        return self.x,self.y,self.theta

    def estimate_controls(self,target_pose,current_pose,v_prev,w_prev):
        '''     
            Input
                previous state: [x_t, y_t, θ_t]
                target_pose : [x_t+1, y_t+1, θ_t+1]
            Output
                controls :[v, w]
            
            w=(θ_t+1 - θ_t)/delta_t
            v=sqrt([(x_t+1 - x_t )/delta_t ]^2+[(y_t+1 -y_t)/delta_t]^2)
        '''  
        # how to calculate the target value value of theta for current pose 
        self.w=w_prev+10*(target_pose.theta - current_pose.theta)/(self.delta_t)
        dx=(target_pose.x- current_pose.x)/(self.delta_t)
        dy=(target_pose.y- current_pose.y)/(self.delta_t)
        self.v=v_prev+10*np.sqrt(dx*dx + dy*dy)

        # check if acceleration is more 
        if (self.v)/self.delta_t >0.288:
            self.v=0.288
        if (self.w)/self.delta_t >5.579:
            self.w=5.579
        
        return self.v,self.w
    

class Map():
    '''  Initialize grid map for Astar path planning '''
    def __init__(self,xrange,yrange,dx,dy,type):
        self.xmin=xrange[0]
        self.xmax=xrange[1]
        self.ymin=yrange[0]
        self.ymax=yrange[1]
        self.w=dx
        self.h=dy
    
        self.make_grid()

    def make_grid(self):
        '''  Build a grid with  cell that contains a landmark as occupied'''
        self.col=int((self.xmax-self.xmin)/self.w)
        self.row=int((self.ymax-self.ymin)/self.h)
        self.cost_map=np.ones((self.row,self.col))
        self.robot_map=np.ones((self.row,self.col))
        
        data=np.loadtxt('ds1/ds1_Landmark_Groundtruth.dat',unpack=True)
        x_o=data[1]
        y_o=data[2]
        
        for i in range(len(x_o)):  
            # if i>4:
            #     break      
            col,row=self.calc_cell(x_o[i],y_o[i])
            self.cost_map[row][col]=1000
            # print('landmark at {},{} has index {},{}'.format(x_o[i],y_o[i],col,row))

            if self.w!=1 :
                self.inflate_obstacles(row,col,self.cost_map)
        
        
        if type=='offline':
            return self.cost_map
        if type=='online':
            return self.robot_map


    def calc_cell(self,x,y):
            '''  calculate indexes in the grid '''
            # print(x,y)
            # x=round(x)
            # y=round(y)
            # print(x,y)
            xindex=int((x-self.xmin)/self.w)
            yindex=int((y-self.ymin)/self.h)
            return [xindex,yindex]
    
    def calc_grid_position(self,xindex,yindex):
            '''  calculate indexes in the grid '''
            
            x=(xindex+self.xmin)*self.w
            y=(yindex+self.ymin)*self.h
            return x,y

    def inflate_obstacles(self,y,x,map):
        '''
          Inflates the amount of space each landmark occupies  in all  directions
        '''
        map[y][x]=1000
        for i in range(0,2):
           
                    map[y+i][x]=1000
                    map[y-i][x]=1000
                    map[y][x+i]=1000
                    map[y][x-i]=1000
                    map[y+i][x+i]=1000
                    map[y-i][x-i]=1000
                    map[y-i][x+i]=1000
                    map[y+i][x-i]=1000
        
        map[y+2][x+1]=1000
        map[y-2][x+1]=1000
        map[y+2][x-1]=1000
        map[y-2][x-1]=1000
        map[y+2][x]=1000
        map[y-2][x]=1000
        return map
       

class Astar():
    '''
        A star path search
        input:
            s: start position [m]   
            g: goal position [m]
            resolution: resolution: grid resolution [m]
            map_type: offline or online observability   

    '''
    def __init__(self,s,g,resolution,map_type,exec=False,problem_no=None):
        self.m=Map(np.array([-2,5]),np.array([-6,6]),resolution,resolution,map_type)
        gx,gy=self.m.calc_cell(g[0],g[1])
        sx,sy=self.m.calc_cell(s[0],s[1])
        self.prev_true_cost=0
        self.s=s
        self.g=g
        self.start=Node(sx,sy,self.m.cost_map[sy][sx])
        
        self.goal=Node(gx,gy,self.m.cost_map[sy][sx])
        self.m.cost_map[sy][sx]= 0
        self.m.cost_map[gy][gx]= 0
        self.exec=exec
        self.controls=inverse_kinematic()
        
        colours = (["white","green","red","blue","yellow","black"])
        # colours = (["white","green","red","blue","yellow",])
        # 1 ,100,200,500,700,1000
        self.cmap = ListedColormap(colours)
        
        if map_type=='offline':
                self.fig=plt.figure()
                self.path_planning(self.m.cost_map)

        if map_type=='online':
            # plt.pcolormesh(self.m.cost_map,cmap=self.cmap)
            self.fig=plt.figure()
            self.start=Node(sx,sy,self.m.robot_map[sy][sx])
            self.goal=Node(gx,gy,self.m.robot_map[gy][gx])
            # print(self.m.cost_map)
            plt.pcolormesh(self.m.robot_map,cmap=self.cmap)
            self.path_planning(self.m.robot_map)
            if self.exec:
                self.path_execution(problem_no)

        
    def heuristic(self,node):
        return np.sqrt((self.goal.x - node.x)**2 + (self.goal.y - node.y)**2)
    
    def calculate_heading(self,node_x,node_y,target_node):
        '''  angle towards the goal '''
        return np.arctan2((target_node.y - node_y),(target_node.x - node_x ))

    def calc_cost(self,node):
        '''  Evaluation function'''
        self.node=self.prev_true_cost + node.cost
        return self.heuristic(node)+ self.prev_true_cost + node.cost

    def path_planning(self,map):
        print('goal is at cell',self.goal.x,self.goal.y)
        print('start is at cell',self.start.x,self.start.y)
        
        open=[]  #stores  nodes for expansions
        visited=[] #stores nodes which we have explored

        # initialize current position as start position 
        open.append([self.calc_cost(self.start),self.start])
        
        i=0
        while (len(open)>0): 
            open.sort(key=lambda x: x[0])
            _,current=open.pop(0)

            visited.append(current)
            print(current.x,current.y)
            self.prev_true_cost=current.cost
            if current.x==self.goal.x and current.y==self.goal.y:
                print('goal found')
                break
            # else expand in neighbours to explore
            neighbors=current.expand()
            for x,y in neighbors:
                if y==-1:
                    continue
                if x<self.m.col and y<self.m.row:
                    if self.m.cost_map[y][x]==1000 :
                        print('obstacle found' ,y,x)
                        map[y][x]=1000
                        # if self.m.w!=1 :
                            # print('map is ',map)
                            # map=self.m.inflate_obstacles(y,x,map)
                            # print('after inflaction ',map)

                    new_node=Node(x,y,map[y][x])

                    if map[y][x]!=1000 :
                        map[y][x]=700
                    # print(map)  clear
                        
                        # print('neighbours ',new_node.x,new_node.y,new_node.cost)
                    if new_node in visited :
                            # print('already visited ')
                            continue
                            
                    open.append((self.calc_cost(new_node),new_node))
                   
                    plt.pcolormesh(map,cmap=self.cmap)

                    plt.title('A star Path planning between start:'+str(self.s) +'goal:'+ str(self.g))
                    plt.xlabel('x(m)')
                    plt.ylabel('y(m)')
                    # plt.savefig('gifs/p2/a2/'+str(i)+'.png')
                    plt.pause(0.00001)

            # print(self.m.cost_map)
            # print(map)
            # break     

            i+=1

        # all the cell in the path 
        path=[]  # for 9 dribving the path generated by the A star

        x=[]
        y=[]
        for cell in visited:
            # print('in path',cell.x,cell.y)
                        # if map[cell.y][cell.x]!=1000:
                            map[cell.y][cell.x]=500
                            x.append(cell.x+0.5*self.m.h)
                            y.append(cell.y+0.5*self.m.h)
                            
        print(x,y)

        

            
        map[self.start.y][self.start.x]=600
        map[self.goal.y][self.goal.x]=200
        
        plt.pcolormesh(map, cmap=self.cmap)




        x=[]
        y=[]
        import csv

        for cell in visited:
            x.append(cell.x+0.05)
            y.append(cell.y+0.05)
            # with open('path.txt', 'wb') as f:
            # csv.writer(f, delimiter=' ').writerows()
        
        
                

    def path_execution(self,problem_no):
        ''' saved planned path and their limits for executing the path '''
        if problem_no==1:
            filepath='path'+str(problem_no)+'txt'
            # can also load it from txt file which saved planned points
            x=[44.05, 43.05, 42.05, 41.05, 40.05, 39.05, 38.05, 37.05,33.05, 32.05, 31.05, 30.05, 29.05, 29.05, 29.05, 29.05, 29.05, 29.05, 29.05]
            y=[24.05, 25.05, 26.05, 27.05, 28.05, 29.05, 30.05, 31.05,34.05, 35.05, 36.05, 37.05, 38.05, 39.05, 40.05, 41.05, 42.05, 43.05, 44.05]
            # plt.xlim(20,60)
            # plt.ylim(20,60)
        elif problem_no ==2:
            x=[69.05, 68.05, 67.05, 66.05, 65.05, 65.05, 64.05, 63.05, 62.05, 61.05, 60.05, 59.05, 58.05, 57.05, 56.05, 55.05, 54.05, 53.05, 52.05, 51.05, 50.05, 49.05, 48.05, 47.05, 46.05, 45.05, 44.05] 
            y=[59.05, 60.05, 61.05, 62.05, 63.05, 64.05, 65.05, 66.05, 67.05, 68.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05, 69.05]
            # plt.xlim(40,70)
            # plt.ylim(50,100)
        elif problem_no==3:
            x=[14.05, 15.05, 16.05, 17.05, 18.05, 19.05, 20.05, 21.05, 22.05, 23.05, 24.05, 25.05, 26.05, 27.05, 26.05, 26.05, 27.05, 28.05, 29.05, 30.05, 31.05, 32.05, 33.05, 34.05, 35.05, 36.05, 37.05, 38.05, 39.05]
            y=[74.05, 75.05, 76.05, 77.05, 78.05, 79.05, 80.05, 81.05, 82.05, 83.05, 84.05, 85.05, 86.05, 87.05, 88.05, 89.05, 90.05, 91.05, 92.05, 93.05, 94.05, 95.05, 96.05, 97.05, 98.05, 99.05, 99.05, 99.05, 99.05]
        current_pose=Pose(x[0],y[0])
        v_prev=0
        w_prev=0
        prev_pose=Pose(0,0,0)
        for i in range(1,len(x)):
                print(i)
                theta=np.arctan2((y[i]- current_pose.y),(x[i] - current_pose.x ))
                target_pose=Pose(x[i],y[i],theta)
                k=0
                print(current_pose.x)
                
                while(abs(current_pose.x-target_pose.x)>0.1):
                    v,w=self.controls.estimate_controls(target_pose,current_pose,v_prev,w_prev)
                    
                    theta=np.arctan2((target_pose.y- current_pose.y),(target_pose.x - current_pose.x ))
                    
                    if current_pose.x-target_pose.x <0:
                        current_pose.x+=v * np.cos(current_pose.theta) * 0.1
                    else:
                        current_pose.x-=v * np.cos(current_pose.theta) * 0.1
                    if prev_pose.y==target_pose.y:  
                        current_pose.x-= v*np.sin(current_pose.theta)*0.1
                    if current_pose.y-target_pose.y <0:
                        current_pose.y+=v * np.sin(current_pose.theta) * 0.1
                    else:
                        current_pose.y-=v * np.sin(current_pose.theta) * 0.1
                    current_pose.theta+=w * 0.1

                    
                
                    
                    # plt.scatter(current_pose.x,current_pose.y,color='blue')
                    if k%50==0:
                        # plt.scatter(current_pose.x,current_pose.y,color='blue')
                        # plt.plot(current_pose.x,current_pose.y,'b->')
                        plt.quiver(current_pose.x,current_pose.y,np.cos(current_pose.theta+np.pi/2),np.cos(current_pose.theta),scale=100)
                    
                    if k>500:
                        break
                
                    k+=1

                prev_pose=target_pose

        for j in range(i,len(x)):
                    print(j)
                    theta=np.arctan2((y[i]- current_pose.y),(x[i] - current_pose.x ))
                    target_pose=Pose(x[i],y[i],theta)
                    k=0
                    print(target_pose)
                    while(abs(current_pose.y-target_pose.y)>0.1):
                        if current_pose.y-target_pose.y <0:
                            current_pose.y+=v * 0.1
                        else:
                            current_pose.y-=v * np.sin(current_pose.theta) * 0.1
                        if current_pose.theta-target_pose.theta <0:
                            current_pose.theta+=w * 0.1
                        else:
                            current_pose.theta-=w * 0.1

                        if k>500:
                            break
                        if k%10==0:
                            # plt.scatter(current_pose.x,current_pose.y,color='blue')
                            # plt.plot(current_pose.x,current_pose.y,'b->')
                            
                            plt.quiver(current_pose.x,current_pose.y,np.cos(current_pose.theta+np.pi/2),np.cos(current_pose.theta),scale=100)
                    
                        k+=1

        




# # reolution 1X1
# a=Astar([ 0.5,-1.5 ],[ 0.5, 1.5 ],1,'offline')
# a=Astar([ 4.5,3.5 ],[ 4.5, -1.5 ],1,'offline')
# a=Astar([-0.5,-5.5 ],[ 1.5,-3.5 ],1,'offline')

a=Astar([ 0.5,-1.5 ],[ 0.5, 1.5 ],1,'online')
# a=Astar([ 4.5,3.5 ],[ 4.5, -1.5 ],1,'online')
# a=Astar([-0.5,-5.5 ],[ 1.5,-3.5 ],1,'online')

# 

# # #resolution 0.1X0.1
#   # not this is partial observability of obstacles 
# a=Astar([ 2.45,-3.55 ],[0.95,-1.55 ],0.1,'online')
# a=Astar([4.9,-0.05],[ 2.45, 0.95],0.1,'online')
# a=Astar([ -0.55,1.45 ],[ 1.95, 3.95 ],0.1,'online')



# # part 9 
# # #resolution 0.1X0.1
# a=Astar([ 2.45,-3.55 ],[0.95,-1.55 ],0.1,'online',True,1)
# a=Astar([4.9,-0.05],[ 2.45, 0.95],0.1,'online',True,2)
# a=Astar([ -0.55,1.45 ],[ 1.95, 3.95 ],0.1,'online',True,3)

# a=Astar([ 0.5,-1.5 ],[ 0.5, 1.5 ],1,'online')
# a=Astar([ 4.5,3.5 ],[ 4.5, -1.5 ],1,'online')
# a=Astar([-0.5,-5.5 ],[ 1.5,-3.5 ],1,'online')


# plt.xlim([20,50])
# plt.ylim([0,60])
plt.show()










