import numpy as np 
import sys
sys.path.insert(0,r"F:\extended_desktop\code")


def deg_to_3space(dest:tuple) -> np.array:
    """
    converts (lat, long) coordinates into 3space with 
    north pole (lat 90, long nan) --> (0,0,1) 
    equator near greenwhich (lat 0, long 0) --> (1,0,0)
      
    so: a triple toward (greenwhich, nepal, north pole)
    """
    assert len(dest) == 2, "destination lat/long misformatted: expect degrees lat long"
    latRad = dest[0]*np.pi/180 
    longRad = dest[1]*np.pi/180

    # equator @ greenwich (1,0,0)
    x = np.cos(longRad)*np.cos(latRad)
    
    # equator @ ~nepal (0,1,0)
    y = np.sin(longRad)*np.cos(latRad)
    
    # north (0,0,1)
    z = np.sin(latRad) 
    
    return np.array([x,y,z])


def magnitude(a:np.array) -> float:
    """get length of a vector"""
    return np.sqrt(sum([c**2 for c in a]))


# function that defines basis vectors for a given lat/long
def lat_long_to_tangent_space_basis_vectors(lat:float, long:float) -> np.array:
    """
    converts (lat, long) coordinates into x,y,z axes of
    the tangent plane (down, north, east).
    """
    
    latRad = lat*np.pi/180 
    longRad = long*np.pi/180
    
    # lat, long determines direction to center
    down = -deg_to_3space( (lat,long) )
    
    # now project out z component of north to establish northward direction vector in tangent plane
    north_normal_to_down = down - np.dot(down,np.array([0,0,1]))
    northward_normal_to_down_as_unit_vect = north_normal_to_down/magnitude(north_normal_to_down)
    north = np.array([np.sin(latRad)*northward_normal_to_down_as_unit_vect[0],
                      np.sin(latRad)*northward_normal_to_down_as_unit_vect[1],
                      np.cos(latRad)])
    
    east = np.cross(down,north) ## and get this part the easy way

    return (down, north, east)


def remapper(src:tuple, dst:tuple, verbose:bool=False) -> np.array:
    """3space compass directions from source to destination"""
    
    src3sp = deg_to_3space(src)
    dst3sp = deg_to_3space(dst)
    pointer = dst3sp-src3sp
    if verbose:
        print(f"{src3sp=}")
        print(f"{dst3sp=}")
        print(f"{pointer=}")
    
    downward, northward, eastward = lat_long_to_tangent_space_basis_vectors(*src)
    if verbose:
        print(f"{downward=}")
        print(f"{northward=}")
        print(f"{eastward=}")
        
    assert all( [(magnitude(v)-1)<(1e-5) for v in [downward, northward, eastward]]  ) 
    
    nComp = np.dot(pointer, northward)
    eComp = np.dot(pointer, eastward)
    dComp = np.dot(pointer, downward)
    if verbose:
        print(f"{nComp=}")
        print(f"{eComp=}")
        print(f"{dComp=}")
    
    ## there's an annoying rotation here to get compass directions
    north_of_east = np.angle( np.complex(eComp,nComp) )
    east_of_north = np.pi/2-north_of_east
    compass = east_of_north
    
    dip = np.arcsin( dComp / magnitude(pointer) ) # angle down
    if verbose:
        print(f"{compass=}")
        print(f"{dip=}")
    
    compassDeg = 360*compass/(2*np.pi)
    dipDeg = 360*dip/(2*np.pi)
    
    return compassDeg, dipDeg

