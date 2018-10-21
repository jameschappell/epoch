"""Python script for automatically generating EPOCH simulations based on an
original FlashForward simulation scheme investigating long term plasma
evolution.

Created by James Chappell.

"""

import numpy as np
import os
import glob
import shutil
import string
import argparse

inputdeck = '''
begin:constant
    samples = 10
   den_norm = 3e22
   beam_dens = 1.101*den_norm
   ppc = 8
   wp = ((den_norm * 1.6e-19 * 1.6e-19) / (9.11e-31 * 8.854e-12))^0.5
   kp = wp / c
   lambdap = (2*pi*c)/wp
   E_0 = 0.511e-3/(c/wp)

   N = 1.25e10 # lowered to avoid blowout regime
   sig_y = (2^0.5) / kp
   sig_x = sig_y
   e_norm = N / ((2*pi)^1.5 * sig_y * sig_y * sig_x)

   energy = 3e9 * 1.6e-19
   emit = 140e-9
   tempy = (energy^2 * emit^2) / (me * c^2 * kb * sig_y^2)
   
   # CONSTANTS
  classic_radius = 2.818e-15
  kJ_mol = 1000/6.022e23
   
   # LASERS
  w = 40*micron			#spatial width described in papers
  sigma_drive = (w / 2^(1/2))	#conversion for gaussian
  sigma_inj = 4 * 2^(1/2) * micron
  lambda0 = 1*micron
  L = 15 / kp
  theta = atan((y_max-y_min)/(2*L))
  a0inj   = 2
  a0drive = 1
  E = 10e-3	#10mJ
  tau = 100e-15 	#100fs
  w0 = 40e-6	#40 microns
  int = (2/pi)*(E/tau)/(w0*w0)
   
   # PLASMA
  yzero = 2 * sigma_drive
  den_depth = 1 / (3.14 * classic_radius * yzero * yzero)

   # BEAM FUNCTIONS
  x_gauss = gauss(x,50e-6,30e-6) 
  y_gauss = gauss(y,0,10e-6)
  #z_gauss = gauss(z,0,0.2e-3)
end:constant


begin:control
  nx = 1200 #?? per lambda_p
  ny = 480 #?? per lambda_p
  #nz = 160

  # maximum number of iterations
  # set to -1 to run until finished
  nsteps = -1

  # final time of simulation
  t_end = 100e-12 + 2e-3/c # xmax

  # size of domain
  x_min = 0.0
  x_max = 1500e-6
  y_min = -300e-6
  y_max = 300e-6
  #z_min = -2e-3
  #z_max = 2e-3

  restart_snapshot_def

  stdout_frequency = 10
end:control


begin:boundaries
  bc_x_min = open		#edit for laser
  bc_x_max = simple_outflow
  bc_y_min = simple_outflow
  bc_y_max = simple_outflow
 # bc_z_min = simple_outflow
 # bc_z_max = simple_outflow
end:boundaries


begin:species
  name = hydrogen
  charge = 1.0
  mass = 1836
  npart = nx * ny * 4
  density = den_norm
end:species

begin:species
  name = beam
  charge = -1.0
  mass = 1
  npart = nx * ny * 8
  density = beam_dens*x_gauss*y_gauss
  drift_x = 1.25 * 5.36e-19
end:species

begin:injector
  boundary = x_min
  species = beam
  t_start_def
  t_end_def
  npart_per_cell = 4
  density = 10*beam_dens*x_gauss*y_gauss
  drift_x = 1.25 * 5.36e-19
  use_flux_maxwellian = T
end:injector
  
begin:dist_fn
  name = x_px
  ndims = 2
  dumpmask = always
  direction1 = dir_x
  direction2 = dir_px
  range1 = (1,1)	# ignored for spatial coordinates
  range2 = (0.1,0.1)
  resolution1 = 1 	# ignored for spatial coordinates
  resolution2 = 1400
  include_species:beam
end:dist_fn

begin:dist_fn
  name = y_py
  ndims = 2
  dumpmask = always
  direction1 = dir_y
  direction2 = dir_py
  range1 = (1,1)        # ignored for spatial coordinates
  range2 = (-0.5 * 5.36e-19,0.5 * 5.36e-19)
  resolution1 = 1       # ignored for spatial coordinates
  resolution2 = 10000
  include_species:beam
end:dist_fn


begin:species
  name = electron
  charge = -1
  mass = 1
  npart = nx*ny*4
  density = density(hydrogen)
end:species

begin:output
  # number of timesteps between output dumps
  dt_snapshot =  0.01e-3/c
  #restart_dump_every = 8
  force_final_to_be_restartable = T
  #time_start = 60*(8*lambdap)/c
  #time_stop = 90*(8*lamdbap)/c 
  # Properties on grid
  grid = always
  ex = always
  ey = always
  #ez = always
  #bx = always
  #by = always
  #bz = always
  number_density = always + species
  temperature = always + species
  distribution_functions = always
end:output

#begin:window
#  move_window = T
#  window_v_x = c
#  window_start_time = 0
#  bc_x_min_after_move = simple_outflow
#  bc_x_max_after_move = simple_outflow
#end:window

'''

subscript = '''
#!/bin/bash -l

#$ -S /bin/bash

#module unload mpi
#module unload compilers
#module load compilers/intel/2017/update1
#module load mpi/intel/2017/update1/intel
module load hdf/5-1.10.2/intel-2018

#$ -l h_rt=24:00:00
#$ -l mem=4G
### -l tmpfs=15G
#$ -N epochplasmarel
#$ -m be
#$ -M james.chappell.17@ucl.ac.uk
work_dir_def
#$ -e ./logs/
#$ -o ./logs/
#$ -pe mpi 200

cd_def

export PATH="/home/ucapjch/Binaries/:$PATH"

gerun /home/ucapjch/Binaries/epoch2d
'''

usedata = '''.'''

def make_environment(resultsdir, delay, length = 50e-6):

    """Edits template files and updates for required values."""

    sg = os.path.join(resultsdir, 'input.deck')
    fh = open(sg, "wb")

    restart_string = '  restart_snapshot = ' + str(delay)
    inputdeck1 = string.replace(inputdeck, 'restart_snapshot_def',
                                       restart_string)

    injector_start_string = '  t_start = ' + str(delay) + 'e-12'
    inputdeck2 = string.replace(inputdeck1, 't_start_def',
                                injector_start_string)

    injector_end_string = '  t_end = ' + str(delay) + 'e-12 + ' + str(length)\
                          + '/c'
    inputdeck3 = string.replace(inputdeck2, 't_end_def', injector_end_string)

    fh.write(inputdeck3)
    fh.close()

    ss = os.path.join(resultsdir, 'sub_script.bash')
    fs = open(ss, "wb")

    work_dir_string = '#$ -wd ' + os.getcwd() + '/' + res_dir
    subscript1 = string.replace(subscript, 'work_dir_def', work_dir_string)

    change_dir_string = 'cd ' + os.getcwd() + '/' + res_dir
    subscript2 = string.replace(subscript1, 'change_dir', change_dir_string)

    fs.write(subscript2)
    fs.close()

    udd = os.path.join(resultsdir, 'USE_DATA_DIRECTORY')
    fd = open(udd, "wb")
    fd.write(usedata)
    fd.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
        This script generates a number of EPOCH simulations.""",
                                     formatter_class=argparse.
                                     RawTextHelpFormatter)

    parser.add_argument('--file', dest='file', default=None,
                        help='''
        This is the path to the directory that contains the output .sdf files 
        which provide the restart files for the subsequent simulations.

        E.g. --file "<path to file>"''')

    parser.add_argument('--start', dest='start', default=None,
                        help='''
        This is the first delay value you want to use within subsequent 
        simulations. Corresponds to the number of the output .sdf file.
        
        E.g. --start 20''')

    parser.add_argument('--end', dest='end', default=None,
                        help='''
        This is the last delay value you want to use within subsequent 
        simulations. Corresponds to the number of the output .sdf file.

        E.g. --end 200''')

    parser.add_argument('--stepsize', dest='stepsize', default=None,
                        help='''
        This is the stepsize between start and end.

        E.g. --stepsize 20''')

    parser.add_argument('--length', dest='length', default=50e-6,
                        help='''
            This is the length of the secondary injected beam in metres.

            E.g. --length 50e-6''')

    arguments = parser.parse_args()

    filename = arguments.file

    start = float(arguments.start)

    end = float(arguments.end)

    step = float(arguments.stepsize)

    length = float(arguments.length)
    # du bist doof
    cwd = os.getcwd()

    delay_values = np.linspace(start, end+1, step)

    for delay in delay_values:

        os.chdir(cwd)

        print "Delay: %s ps" % (delay)
        res_dir = str(delay)
        print "Making directory: ", res_dir
        if os.path.isdir(res_dir) is False:
            os.mkdir(res_dir)
        else:
            print "Directory %s already exists. Stopping." % (res_dir)
            break
        os.chdir(filename)
        copyname = format(delay, "04") + '.sdf'
        shutil.copyfile(copyname, cwd + '/' + res_dir)

        make_environment(res_dir, delay, length)
        os.chdir(cwd + '/' + res_dir)
        os.mkdir('logs')
        run_command = "qsub sub_script.bash"
        print run_command
        #os.system(run_command)




