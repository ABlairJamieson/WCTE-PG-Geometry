#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

from Geometry.WCD import WCD



# From drawing of mPMT module some global constant offests
mpmt_to_dome_center = -80.1 # mm
mpmt_to_matrix = 154.8 #mm 
mpmt_matrix_radius = 297.2 # Note mPMT is recessed... matrix goes out to 322.0 # mm
mpmt_dome_inner_radius = 332.0 #mm
mpmt_dome_outer_radius = 342.0 #mm
pmt_radius = 45.0 # mm

# make an instance of Dean's geometry of WCTE
wcte = WCD('wcte', kind='WCTE')

# In[13]:

def get_camera_pos_dir():
    '''
    Function to return the location and pointing of the eight cameras in WCTE coordinate system.
    WCTE (0,0,0) is at the center of the tank in z,x and at the center of the beam height in y.
    For now this isn't using Dean's geometry code, but will be updated when he adds cameras.
    
    Returns:
    (cam_x, cam_y, cam_z, cam_dirx, cam_diry, cam_dirz)
    
    cam_x  is an np.array of x locations of the eight cameas (cam_x[0] is x of first camera...)
    cam_y  is an np.array of y locations
    cam_z  is an np.array of z locations
    cam_dxx is an np.array of the x component of the horizontal along pixels of each camera
    cam_dxy is an np.array of the y component of the horizontal along pixels of each camera
    cam_dxz is an np.array of the z component of the horizontal along pixels of each camera
    cam_dzx is an np.array of the x component of the normal vector of the pointing of each camera
    cam_dzy is an np.array of the y component of the normal vector of the pointing of each camera
    cam_dzz is an np.array of the z component of the normal vector of the pointing of each camera
    '''
    # by looking at WCTE drawing, pick off locations of cameras (approximately!)
    # z is beam direction
    # y is vertical
    # x is remaining coordinate
    #cam_x = np.array([ 1933.8, 1933.8,  1933.8,  1933.8, -1933.8, -1933.8, -1933.8, -1933.8 ])
    #cam_z = np.array([ 1933.8, 1933.8, -1933.8, -1933.8,  1933.8,  1933.8, -1933.8, -1933.8 ])
    #cam_x = np.divide(cam_x,2.0) # diameter to radius
    #cam_z = np.divide(cam_z,2.0)
    #cam_y = np.array([ 1728.0, -780.0,  1728.0,  -780.0,  1728.0,  -728.0,  1728.0,  -728.0 ])

    #iroot3 = 1./np.sqrt(3.0)
    #cam_dirx = np.array([-iroot3, -iroot3, -iroot3, -iroot3, iroot3, iroot3, iroot3, iroot3])
    #cam_diry = np.array([-iroot3, -iroot3,  iroot3,  iroot3, -iroot3, -iroot3, iroot3, iroot3])
    #cam_dirz = np.array([-iroot3, iroot3, -iroot3, iroot3, -iroot3, iroot3, -iroot3, iroot3])
    cam_x = []
    cam_y = []
    cam_z = []
    cam_dxx = []
    cam_dxy = []
    cam_dxz = []
    cam_dzx = []
    cam_dzy = []
    cam_dzz = []

    for icam, camera in enumerate(wcte.cameras):
        p = camera.get_placement('design')
        location, direction_x, direction_z = p['location'], p['direction_x'], p['direction_z']
        cam_x.append( location[0] )
        cam_y.append( location[1] )
        cam_z.append( location[2] )
        cam_dxx.append( direction_x[0] )
        cam_dxy.append( direction_x[1] )
        cam_dxz.append( direction_x[2] )
        cam_dzx.append( direction_z[0] )
        cam_dzy.append( direction_z[1] )
        cam_dzz.append( direction_z[2] )
    
    cam_x = np.array( cam_x )
    cam_y = np.array( cam_y )
    cam_z = np.array( cam_z )
    cam_dxx = np.array( cam_dxx )
    cam_dxy = np.array( cam_dxy )
    cam_dxz = np.array( cam_dxz )
    cam_dzx = np.array( cam_dzx )
    cam_dzy = np.array( cam_dzy )
    cam_dzz = np.array( cam_dzz )

    return (cam_x, cam_y, cam_z, cam_dxx, cam_dxy, cam_dxz, cam_dzx, cam_dzy, cam_dzz)
    


# example usage:
def test_get_camera_pos_dir():
    cam_x, cam_y, cam_z, cam_dirx, cam_diry, cam_dirz = get_camera_pos_dir()


    # test camera positions
    ax = plt.figure(figsize=(3,3),dpi=200).add_subplot(projection='3d')

    ax.scatter( cam_x, cam_y, cam_z,s=2,color='r')
    ax.quiver( cam_x, cam_y, cam_z, cam_dirx, cam_diry, cam_dirz, length=200,normalize=True)
    plt.show()


# In[4]:



# In[15]:

def get_mpmt_dome_blacks_pos_dir():
    '''
    Function to return the location and pointing of the mPMT module dome centers and blacksheet 
    in WCTE coordinate system. WCTE (0,0,0) is at the center of the tank in z,x and at the center 
    of the beam height in y.
    
    Returns:
    (dome_x, dome_y, dome_z, blacks_x, blacks_y, blacks_z, 
       u_x, u_y, u_z, w_x, w_y, w_z ) 
       
    dome_x  is an np.array of x locations of the mPMTs (dome_x[0] is x of center of mPMT dome...)
    dome_y  is an np.array of y locations " 
    dome_z  is an np.array of z locations " 
    blacks_x  is an np.array of x locations of the blacksheet (blacks_x[0] is x of center of blacksheet...)
    blacks_y  is an np.array of y locations "
    blacks_z  is an np.array of z locations " 
    u_x is an np.array of x component of normal vector of pointing of each mPMT
    u_y is an np.array of y component of normal vector of pointing of each mPMT
    u_z is an np.array of z component of normal vector of pointing of each mPMT
    w_x is an np.array of x component of normal vector in a plane perp to pointing of mPMT
    w_y is an np.array of y component of normal vector in a plane perp to pointing of mPMT
    w_z is an np.array of z component of normal vector in a plane perp to pointing of mPMT
'''
    
    mpmt_origins = []  # center of backplate of mpmt module
    mpmt_z_vecs = []   # z_vec is the pointing, x_vec is in the plane of the module
    mpmt_x_vecs = []

    for i_mpmt,mpmt in enumerate(wcte.mpmts):  
        p = mpmt.get_placement('design')
        location, direction_x, direction_z = p['location'], p['direction_x'], p['direction_z']
        # lists to show mPMT coordinate systems
        mpmt_z_vec = np.array(direction_z)
        mpmt_x_vec = np.array(direction_x)
   
        mpmt_origins.append(location)
        mpmt_z_vecs.append(mpmt_z_vec)
        mpmt_x_vecs.append(mpmt_x_vec)

    mpmt_origins = np.array(mpmt_origins)  # center of backplate of mpmt module
    mpmt_z_vecs = np.array(mpmt_z_vecs)   # z_vec is the pointing, x_vec is in the plane of the module
    mpmt_x_vecs = np.array(mpmt_x_vecs)

    u_dirx = mpmt_z_vecs.T[0]
    u_diry = mpmt_z_vecs.T[1]
    u_dirz = mpmt_z_vecs.T[2]
    
    w_dirx = mpmt_x_vecs.T[0]
    w_diry = mpmt_x_vecs.T[1]
    w_dirz = mpmt_x_vecs.T[2]
    
    mpmt_x = mpmt_origins.T[0]
    mpmt_y = mpmt_origins.T[1]
    mpmt_z = mpmt_origins.T[2]
    
    # Calculate dome center (assumes mPMT location is center of baseplate)
    # dome facing is mpmt_dirx,mpmt_diry,mpmt_dirz
    dome_x = mpmt_x + u_dirx * mpmt_to_dome_center
    dome_y = mpmt_y + u_diry * mpmt_to_dome_center
    dome_z = mpmt_z + u_dirz * mpmt_to_dome_center

    # Calculate center of a blacksheet positioned relative to center of dome
    # blacksheet facing is same as mPMT module facing
    blacks_x = dome_x + u_dirx * mpmt_to_matrix
    blacks_y = dome_y + u_diry * mpmt_to_matrix
    blacks_z = dome_z + u_dirz * mpmt_to_matrix
    
    return (dome_x, dome_y, dome_z, blacks_x, blacks_y, blacks_z, u_dirx, u_diry, u_dirz, w_dirx, w_diry, w_dirz )


# In[18]:

def test_get_mpmt_dome_blacks_pos_dir():
    dome_x, dome_y, dome_z, blacks_x, blacks_y, blacks_z, u_x, u_y, u_z, w_x, w_y, w_z = get_mpmt_dome_blacks_pos_dir()

    # Check dome and blacksheet values using a plot
    fig=plt.figure(figsize=(4,4),dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.quiver(dome_x,dome_y, dome_z, u_x, u_y, u_z, length=200,normalize=True,color='b')
    ax.scatter(dome_x,dome_y,dome_z,'.',s=2, label='dome',color='r')
    ax.scatter(blacks_x,blacks_y,blacks_z,'.',s=2, label='blacks',color='g')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()


# In[19]:


def get_pmt_loc_dir():
    '''
    Function to return the location and pointing of the PMTs that are inside mPMT modules
    in WCTE coordinate system.
    WCTE (0,0,0) is at the center of the tank in z,x and at the center of the beam height in y.
    For now this isn't using Dean's geometry code, but will be updated when he adds cameras.
    
    Returns:
    (pmt_x, pmt_y, pmt_z, pmt_dirx, pmt_diry, pmt_dirz)
    
    pmt_x  is an np.array of x locations of the PMTs (pmt_x[0] is x of first pmt...)
    pmt_y  is an np.array of y locations
    pmt_z  is an np.array of z locations
    pmt_dirx is an np.array of the x component of the normal vector of the pointing of each pmt
    pmt_diry is an np.array of the y component of the normal vector of the pointing of each pmt
    pmt_dirz is an np.array of the z component of the normal vector of the pointing of each pmt
    '''
    
    pmt_origins = []
    pmt_z_vecs = []
    pmt_x_vecs = []

    for i_mpmt,mpmt in enumerate(wcte.mpmts):  
        for i_pmt, pmt in enumerate(mpmt.pmts):
            p = pmt.get_placement('design')
            location, direction_x, direction_z = p['location'], p['direction_x'], p['direction_z']
            pmt_z_vec = np.array(direction_z)
            pmt_x_vec = np.array(direction_x)
            pmt_origins.append(location)
            pmt_z_vecs.append(pmt_z_vec)
            pmt_x_vecs.append(pmt_x_vec)
        
    pmt_origins = np.array(pmt_origins)
    pmt_z_vecs = np.array(pmt_z_vecs)
    pmt_x_vecs = np.array(pmt_x_vecs)

    pmt_x= pmt_origins.T[0]
    pmt_y = pmt_origins.T[1]
    pmt_z = pmt_origins.T[2]
        
    pmt_dirx = pmt_z_vecs.T[0]
    pmt_diry = pmt_z_vecs.T[1]
    pmt_dirz = pmt_z_vecs.T[2]
    
    # Above is center of cathode... shift to center of pmtCalculate dome center 
    pmt_x = pmt_x - pmt_dirx * pmt_radius
    pmt_y = pmt_y - pmt_diry * pmt_radius
    pmt_z = pmt_z - pmt_dirz * pmt_radius

    return (pmt_x, pmt_y, pmt_z, pmt_dirx, pmt_diry, pmt_dirz)


# In[20]:

def test_get_pmt_loc_dir():
    pmt_x, pmt_y, pmt_z, pmt_dirx, pmt_diry, pmt_dirz =  get_pmt_loc_dir()

    fig=plt.figure(figsize=(4,4),dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pmt_x,pmt_y,pmt_z,'.',s=2,label='pmt')
    ax.quiver(pmt_x, pmt_y, pmt_z, pmt_dirx, pmt_diry, pmt_dirz, length=100,normalize=True,color='r',linewidth=0.5)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()


# In[25]:




def get_led_loc_dir():
    '''
    Function to return the location and pointing of the LEDs that are inside mPMT modules
    in WCTE coordinate system.
    WCTE (0,0,0) is at the center of the tank in z,x and at the center of the beam height in y.
    For now this isn't using Dean's geometry code, but will be updated when he adds cameras.
    
    Returns:
    (led_x, led_y, led_z, led_dirx, led_diry, led_dirz)
    
    led_x  is an np.array of x locations of the LEDs (led_x[0] is x of first LED...)
    led_y  is an np.array of y locations
    led_z  is an np.array of z locations
    led_ux is an np.array of the x component of the normal vector of the pointing of each pmt
    led_uy is an np.array of the y component of the normal vector of the pointing of each pmt
    led_uz is an np.array of the z component of the normal vector of the pointing of each pmt
    led_wx
    led_wy
    led_wz
    led_label is a label mmm-l for mpmt number mmm and led l
    '''

    led_origins = []
    led_z_vecs = []
    led_x_vecs = []
    led_labels = []

    for i_mpmt,mpmt in enumerate(wcte.mpmts):          
        for i_led, led in enumerate(mpmt.leds):
            #print('kind=',led.kind)
            #if led.kind == 'LD':
                p = led.get_placement('design')
                location, direction_x, direction_z = p['location'], p['direction_x'], p['direction_z']
                x_vec = np.array(direction_x)
                z_vec = np.array(direction_z)
                led_origins.append(location)
                led_x_vecs.append(x_vec)
                led_z_vecs.append(z_vec)
                led_labels.append( f'{i_mpmt:03d}-{i_led}' )
    led_origins = np.array(led_origins)
    led_z_vecs = np.array(led_z_vecs)
    led_x_vecs = np.array(led_x_vecs)
    led_x = led_origins.T[0]
    led_y = led_origins.T[1]
    led_z = led_origins.T[2]
    led_ux = led_z_vecs.T[0]
    led_uy = led_z_vecs.T[1]
    led_uz = led_z_vecs.T[2]
    led_wx = led_x_vecs.T[0]
    led_wy = led_x_vecs.T[1]
    led_wz = led_x_vecs.T[2]
    return (led_x, led_y, led_z, led_ux, led_uy, led_uz, led_wx, led_wy, led_wz, led_labels)


# In[26]:

def test_get_led_loc_dir():
    led_x, led_y, led_z, led_dirx, led_diry, led_dirz, led_wx, led_wy, led_wz, labels = get_led_loc_dir()


    fig=plt.figure(figsize=(4,4),dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.quiver( led_x, led_y, led_z, led_dirx, led_diry, led_dirz, length=100,normalize=True,color='r',linewidth=0.5)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

