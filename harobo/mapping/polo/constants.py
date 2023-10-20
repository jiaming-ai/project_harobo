VOXEL_HEIGHT = 34
class MapConstants:
    OBSTACLE_MAP = 0
    EXPLORED_MAP = 1 # 2d projection of all height channels
    CURRENT_LOCATION = 2 
    VISITED_MAP = 3 # only floor height
    BEEN_CLOSE_MAP = 4 # closely explored
    PROBABILITY_MAP = 5 # Probability of goal object being at location (i, j)
    VOXEL_START = 6 # Start of voxel height channels
    VOXEL_END = VOXEL_START + VOXEL_HEIGHT
    END_REC_START = VOXEL_END # start of end receptacle channels 
    END_REC_END = END_REC_START + VOXEL_HEIGHT
    
    NON_SEM_CHANNELS =  END_REC_END # Number of non-semantic channels at the start of maps +  voxel height
    SEMANTIC_START = NON_SEM_CHANNELS # start of semantic channels
# class ProbabilisticMapConstants:
#     NON_SEM_CHANNELS = 5  # Number of non-semantic channels at the start of maps
#     OBSTACLE_MAP = 0
#     EXPLORED_MAP = 1
#     CURRENT_LOCATION = 2
#     VISITED_MAP = 3
#     BEEN_CLOSE_MAP = 4
