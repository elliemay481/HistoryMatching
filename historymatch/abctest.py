from abc import ABC, abstractmethod

class Emulator(ABC):

    @abstractmethod
    def __init__(self, speed):
        self.__speed = speed

    @property
    @abstractmethod
    def speed(self):
        pass

    @speed.setter
    @abstractmethod
    def method(self, str):
        pass

    @abstractmethod
    def simulator(self):
        pass

    @abstractmethod
    def land(self):
        print("All checks completed")

class GP(Emulator):

    def __init__(self, speed):
        self.__speed = speed

    @property
    def speed(self):
        return self.__speed

    @speed.setter
    def method(self, str):
        self.__method = 'GP'

    def simulate(self, x):
        print("My jet is flying")

    def land(self):
        super().land()
        print("My jet has landed")




'''
class HistoryMatch(object):
    def __init__(self, x):
        self.x = x

    def simulator(self):
        raise NotImplementedError("Please Implement this method")

    def simulate(self):
        return self.simulator()

    def get_behaviour(self):
        return self.behaviour

    def get_face(self):
        return self.face


class Simulator(object):

    def set_simulator(self, model):
        #raise NotImplementedError("Please Implement this method")
        self.simulator = model
        HistoryMatch.simulator =model

    def simulate(self):
        return self.simulator()'''

    
class HistoryMatch(): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
        self.simulator = None
  
    def simulate(self, theta):
        if self.simulator is None:
            raise NotImplementedError("Simulator not defined")
        else:
            return self.simulator(theta)

        
  
class Simulator(HistoryMatch): 
    def __init__(self, class_a): 
        #self.x = class_a.x 
        #self.y = class_a.y
        self.class_a = class_a

    def set_simulator(self, model):
        self.class_a.simulator = model
  

    