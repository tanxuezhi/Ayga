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
## theYore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are theYore        ##
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

######################################################################
## This code generates the Figure 4 of article                      ##
## "Multivariate stochastic bias correction with optimal transport" ##
## by Robin and al, DOI: 10.5194/hess-2018-281                      ##
######################################################################

################
## Librairies ##
################

import sys,os
import numpy as np
import Apyga.dynamic.continuous as apydc
import Apyga.stats.bc as apysbc


################
## Matplotlib ##
################

import matplotlib as mpl
mpl.use( "Qt5Agg" )
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

mpl.rcParams['font.size'] = 40
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text',usetex=True)
plt.rcParams['text.latex.unicode'] = True


###########
## Class ##
###########

class Dataset:
	def __init__( self , load = False ):
		
		## Initialize Lorenz84 and time axis
		l84  = apydc.Lorenz84( F = lambda t : 9.5 + 10 * apydc.Lorenz84.TimeForcing.linear( t , tcc = 6 * 73 ) )
		self.t    = np.arange( 0 , 7 * 73 , 0.005 )
		self.size = int(len(self.t)/7)
		self.m = np.array( [1,2,3] )
		self.S = np.array( [ 1.22 , 0. , 0. , -0.41 , 1.04 , 0. , -0.41 , 0.56 , 0.52 ] ).reshape( (3,3) )
		
		orbit = l84.orbit(self.t)[(5 * self.size):,:]
		
		self.Y0 = orbit[:self.size,:]
		self.Y1 = orbit[self.size:,:]
		self.X0 = np.apply_along_axis( lambda x : np.dot(self.S,x+self.m) , 1 , self.Y0 )
		self.X1 = np.apply_along_axis( lambda x : np.dot(self.S,x+self.m) , 1 , self.Y1 )
		
		self.uX0QM   = None
		self.uX1CDFt = None
		self.uX0OTC  = None
		self.uX1dOTC = None


###############
## Fonctions ##
###############

def add_parameters( ax ):
	xLim = (-0.5,4)
	yLim = (-2,4)
	zLim = (-2,4)
	ax.set_xlabel( r"$x$" )
	ax.set_ylabel( r"$y$" )
	ax.set_zlabel( r"$z$" )
	ax.set_xlim( xLim )
	ax.set_ylim( yLim )
	ax.set_zlim( zLim )
	ax.set_xticks( [0,2,4] )
	ax.set_yticks( [-2,0,2,4] )
	ax.set_zticks( [-2,0,2,4] )
	return ax

def plot( data ):
	X0 = data.X0
	X1 = data.X1
	Y0 = data.Y0
	Y1 = data.Y1
	uX0 = data.uX0OTC
	uX1 = data.uX1dOTC
	uX0QM = data.uX0QM
	uX1CDFt = data.uX1CDFt
	
	fig = plt.figure( figsize = (30,20) )
	
	## Calibration
	ax = fig.add_subplot( 2 , 3 , 1 , projection = "3d" )
	ax = add_parameters(ax)
	ax.plot( Y0[:,0] , Y0[:,1] , Y0[:,2] , color = "blue" , linestyle = "" , marker = "." )
	ax.plot( X0[:,0] , X0[:,1] , X0[:,2] , color = "red" , linestyle = "" , marker = "." )
	ax.text( -1.5 , 0 , 5 , r"$\mathrm{(a)}$" , fontsize = 50 )
	ax.text( 2 , -2 , -2 , r"$\mathbf{Y}^0$" , fontsize = 50 )
	ax.text( 3.3 , 4 , 0 , r"$\mathbf{X}^0$" , fontsize = 50 )

	ax = fig.add_subplot( 2 , 3 , 2 , projection = "3d" )
	ax = add_parameters(ax)
	ax.plot( uX0[:,0] , uX0[:,1] , uX0[:,2] , color = "green" , linestyle = "" , marker = "." )
	ax.plot( X0[:,0] , X0[:,1] , X0[:,2] , color = "red" , linestyle = "" , marker = "." )
	ax.text( -1.5 , 0 , 5 , r"$\mathrm{(b)}$" , fontsize = 50 )
	ax.text( 2 , -2 , -2 , r"$\mathbf{Z}^0$" , fontsize = 50 )
	ax.text( 3.3 , 4 , 0 , r"$\mathbf{X}^0$" , fontsize = 50 )
	ax.set_title( r"$\mathrm{OTC}$" )

	ax = fig.add_subplot( 2 , 3 , 3 , projection = "3d" )
	ax = add_parameters(ax)
	ax.plot( uX0QM[:,0] , uX0QM[:,1] , uX0QM[:,2] , color = "green" , linestyle = "" , marker = "." )
	ax.plot( X0[:,0] , X0[:,1] , X0[:,2] , color = "red" , linestyle = "" , marker = "." )
	ax.text( -1.5 , 0 , 5 , r"$\mathrm{(c)}$" , fontsize = 50 )
	ax.text( 2 , -2 , -2 , r"$\mathbf{Q}^0$" , fontsize = 50 )
	ax.text( 3.3 , 4 , 0 , r"$\mathbf{X}^0$" , fontsize = 50 )
	ax.set_title( r"$\mathrm{Quantile\ Mapping}$" )

	## Projection
	ax = fig.add_subplot( 2 , 3 , 4 , projection = "3d" )
	ax = add_parameters(ax)
	ax.plot( Y1[:,0] , Y1[:,1] , Y1[:,2] , color = "blue" , linestyle = "" , marker = "." )
	ax.plot( X1[:,0] , X1[:,1] , X1[:,2] , color = "red" , linestyle = "" , marker = "." )
	ax.text( -1.5 , 0 , 5 , r"$\mathrm{(d)}$" , fontsize = 50 )
	ax.text( 2 , -2 , -2 , r"$\mathbf{Y}^1$" , fontsize = 50 )
	ax.text( 3.3 , 4 , 0 , r"$\mathbf{X}^1$" , fontsize = 50 )
	
	ax = fig.add_subplot( 2 , 3 , 5 , projection = "3d" )
	ax = add_parameters(ax)
	ax.plot( uX1[:,0] , uX1[:,1] , uX1[:,2] , color = "green" , linestyle = "" , marker = "." )
	ax.plot( X1[:,0] , X1[:,1] , X1[:,2] , color = "red" , linestyle = "" , marker = "." )
	ax.text( -1.5 , 0 , 5 , r"$\mathrm{(e)}$" , fontsize = 50 )
	ax.text( 2 , -2 , -2 , r"$\mathbf{Z}^1$" , fontsize = 50 )
	ax.text( 3.3 , 4 , 0 , r"$\mathbf{X}^1$" , fontsize = 50 )
	ax.set_title( r"$\mathrm{dOTC}$" )

	ax = fig.add_subplot( 2 , 3 , 6 , projection = "3d" )
	ax = add_parameters(ax)
	ax.plot( uX1CDFt[:,0] , uX1CDFt[:,1] , uX1CDFt[:,2] , color = "green" , linestyle = "" , marker = "." )
	ax.plot( X1[:,0] , X1[:,1] , X1[:,2] , color = "red" , linestyle = "" , marker = "." )
	ax.text( -1.5 , 0 , 5 , r"$\mathrm{(f)}$" , fontsize = 50 )
	ax.text( 2 , -2 , -2 , r"$\mathbf{Q}^1$" , fontsize = 50 )
	ax.text( 3.3 , 4 , 0 , r"$\mathbf{X}^1$" , fontsize = 50 )
	ax.set_title( r"$\mathrm{CDF-}t$" )
	
	plt.tight_layout()
	plt.savefig( "Robin_and_al_Multivariate_BC_OTC_dOTC_Fig4.png" )


##########
## main ##
##########


if __name__ == "__main__":
	
	## Data
	data = Dataset()
	
	## Univariate bias correction of calibration period with quantile mapping
	qm = apysbc.QM()
	qm.fit( data.Y0 , data.X0 )
	data.uX0QM = qm.predict( data.X0 )
	
	## Univariate bias correction of projection period with CDFt
	cdft = apysbc.CDFt()
	cdft.fit( data.Y0 , data.X0 , data.X1 )
	data.uX1CDFt = cdft.predict( data.X1 )
	
	## Multivariate bias correction of calibration period with OTC
	otc = apysbc.OTC()
	otc.fit( data.Y0 , data.X0 )
	data.uX0OTC = otc.predict( data.X0 )
	
	## Multivariate bias correction of projection period with dOTC
	dotc = apysbc.dOTC( cov_factor = "cholesky" )
	dotc.fit( data.Y0 , data.X0 , data.X1 )
	data.uX1dOTC = dotc.predict( data.X1 )
	
	## Plot figure
	plot(data)
	
	print("Done")


