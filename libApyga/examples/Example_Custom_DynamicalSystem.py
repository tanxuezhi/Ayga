# -*- coding: utf-8 -*-

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## This software is a computer program whose purpose is to study the            ##
## statistics of dynamical systems, statisticaly correct in uni/multivariate    ##
## the bias of dataset by applying the optimal transport to sparse histograms.  ##
##                                                                              ##
## This software is governed by the CeCILL-C license under French law and       ##
## abiding by the rules of distribution of free software.  You can  use,        ##
## modify and/ or redistribute the software under the terms of the CeCILL-C     ##
## license as circulated by CEA, CNRS and INRIA at the following URL            ##
## "http://www.cecill.info".                                                    ##
##                                                                              ##
## As a counterpart to the access to the source code and  rights to copy,       ##
## modify and redistribute granted by the license, users are provided only      ##
## with a limited warranty  and the software's author,  the holder of the       ##
## economic rights,  and the successive licensors  have only  limited           ##
## liability.                                                                   ##
##                                                                              ##
## In this respect, the user's attention is drawn to the risks associated       ##
## with loading,  using,  modifying and/or developing or reproducing the        ##
## software by the user in light of its specific status of free software,       ##
## that may mean  that it is complicated to manipulate,  and  that  also        ##
## therefore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are therefore        ##
## encouraged to load and test the software's suitability as regards their      ##
## requirements in conditions enabling the security of their systems and/or     ##
## data to be ensured and,  more generally, to use and operate it in the        ##
## same conditions as regards security.                                         ##
##                                                                              ##
## The fact that you are presently reading this means that you have had         ##
## knowledge of the CeCILL-C license and that you accept its terms.             ##
##                                                                              ##
##################################################################################
##################################################################################

##################################################################################
##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique servant à étudier les statistiques ##
## des systèmes dynamiques, corriger statistiquement en uni/multivarié des      ##
## données en appliquant le transport optimal à des histogrammes creux.         ##
##                                                                              ##
## Ce logiciel est régi par la licence CeCILL-C soumise au droit français et    ##
## respectant les principes de diffusion des logiciels libres. Vous pouvez      ##
## utiliser, modifier et/ou redistribuer ce programme sous les conditions       ##
## de la licence CeCILL-C telle que diffusée par le CEA, le CNRS et l'INRIA     ##
## sur le site "http://www.cecill.info".                                        ##
##                                                                              ##
## En contrepartie de l'accessibilité au code source et des droits de copie,    ##
## de modification et de redistribution accordés par cette licence, il n'est    ##
## offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,    ##
## seule une responsabilité restreinte pèse sur l'auteur du programme, le       ##
## titulaire des droits patrimoniaux et les concédants successifs.              ##
##                                                                              ##
## A cet égard  l'attention de l'utilisateur est attirée sur les risques        ##
## associés au chargement,  à l'utilisation,  à la modification et/ou au        ##
## développement et à la reproduction du logiciel par l'utilisateur étant       ##
## donné sa spécificité de logiciel libre, qui peut le rendre complexe à        ##
## manipuler et qui le réserve donc à des développeurs et des professionnels    ##
## avertis possédant  des  connaissances  informatiques approfondies.  Les      ##
## utilisateurs sont donc invités à charger  et  tester  l'adéquation  du       ##
## logiciel à leurs besoins dans des conditions permettant d'assurer la         ##
## sécurité de leurs systèmes et ou de leurs données et, plus généralement,     ##
## à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.           ##
##                                                                              ##
## Le fait que vous puissiez accéder à cet en-tête signifie que vous avez       ##
## pris connaissance de la licence CeCILL-C, et que vous en avez accepté les    ##
## termes.                                                                      ##
##                                                                              ##
##################################################################################
##################################################################################

##########
## Path ##
##########

import sys,os
pathApyga = os.path.join( os.path.abspath( os.path.dirname( __file__ ) ) , ".." )
sys.path.append(pathApyga)


###############
## Libraries ##
###############

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
import Apyga.dynamic.continuous as apydc


###########
## Class ##
###########

## Custom continuous dynamical system, based of Apyga.dynamic.continuous.DiffDynSyst
## Written from https://matplotlib.org/gallery/animation/double_pendulum_animated_sgskip.html
class DoublePendulum(apydc.DiffDynSyst):
	def __init__( self , G = 9.8 , L1 = 1.0 , L2 = 1.0 , M1 = 1.0 , M2 = 1.0 , size = 1 ):
		apydc.DiffDynSyst.__init__( self , 4 , size , np.array( [ [ - np.pi / 2 , 0. , - np.pi , 0. ] , [ np.pi / 2 , 1. , np.pi , 1. ] ] ) )
		self.G = G    # Gravity constant
		self.L1 = L1  # Length of pendulum 1
		self.L2 = L2  # Mass of pendulum 1
		self.M1 = M1  # Length of pendulum 2
		self.M2 = M2  # Mass of pendulum 2

	def _equation( self , X , t ):
		## Don't panic! It is "just" the equation of double pendulum
		dX = np.zeros_like(X)
		
		tmp0 = X[self._i[2]] - X[self._i[0]]
		tmp1 = ( self.M1 + self.M2 ) * self.L1 - self.M2 * self.L1 * np.power( np.cos(tmp0) , 2 )
		tmp2 = tmp1 * self.L2 / self.L1

		dX[self._i[0]] = X[self._i[1]]
		dX[self._i[1]] = ( self.M2 * self.L1 * np.power( X[self._i[1]] , 2 ) * np.sin(tmp0) * np.cos(tmp0) +
		                   self.M2 * self.G * np.sin(X[self._i[2]]) * np.cos(tmp0) +
								 self.M2 * self.L2 * np.power( X[self._i[3]] , 2 ) * np.sin(tmp0) -
								 ( self.M1 + self.M2 ) * self.G * np.sin( X[self._i[0]] ) ) / tmp1
		dX[self._i[2]] = X[self._i[3]]
		dX[self._i[3]] = ( - self.M2 * self.L2 * np.power( X[self._i[3]] , 2 ) * np.sin(tmp0) * np.cos(tmp0) +
		                  ( self.M1 + self.M2 ) * self.G * np.sin(X[self._i[0]]) * np.cos(tmp0) -
								( self.M1 + self.M2 ) * self.L1 * np.power( X[self._i[1]] , 2 ) * np.sin(tmp0) - 
								( self.M1 + self.M2 ) * self.G * np.sin(X[self._i[2]]) ) / tmp2

		return dX


##########
## main ##
##########

if __name__ == "__main__":
	
	## Build DoublePendulum
	dp = DoublePendulum()
	
	## Because it is based of Apyga.dynamic.continuous.DiffSynSyst, we can use orbit functions, and can create snapshot
	time = np.arange( 0 , 20 , 0.05 )
	orbit = dp.orbit( time )
	
	## Plot
	fig = plt.figure( figsize = (15,5) )

	ax = fig.add_subplot(1,3,1)
	ax.plot( time , orbit[:,0] , color = "red" , label = "Pendulum 1" )
	ax.plot( time , orbit[:,2] , color = "blue" , label = "Pendulum 2" )
	ax.set_xlabel( "time" )
	ax.set_ylabel( "Angle" )
	ax.legend( loc = "lower right" )

	ax = fig.add_subplot(1,3,2)
	ax.plot( time , orbit[:,1] , color = "red" )
	ax.plot( time , orbit[:,3] , color = "blue" )
	ax.set_xlabel( "time" )
	ax.set_ylabel( "Angular Velocity" )
	

	## Animation
	x1 = dp.L1 * np.sin( orbit[:,0] )
	y1 = - dp.L1 * np.cos( orbit[:,0] )
	
	x2 = dp.L2 * np.sin( orbit[:,2] ) + x1
	y2 = - dp.L2 * np.cos( orbit[:,2] ) + y1

	ax = fig.add_subplot( 1 , 3 , 3 , autoscale_on = False , xlim=(-2, 2) , ylim=(-2, 2) )
	ax.set_aspect('equal')
	ax.grid()
	
	line, = ax.plot( [] , [] , 'o-' , linewidth = 2 )
	time_template = 'time = %.1fs'
	time_text = ax.text( 0.05 , 0.9 , '' , transform = ax.transAxes )
	
	def init():
		line.set_data( [] , [] )
		time_text.set_text('')
		return line,time_text
	
	
	def animate(i):
		thisx = [ 0 , x1[i] , x2[i] ]
		thisy = [ 0 , y1[i] , y2[i] ]
		line.set_data( thisx , thisy )
		time_text.set_text( time_template % ( i * 0.05 ) )
		return line,time_text
	
	ani = animation.FuncAnimation( fig , animate , np.arange( 1 , len(time) ) , interval = 25 , blit = True , init_func = init )
	
	plt.tight_layout()
	plt.show()
	print("Done")
