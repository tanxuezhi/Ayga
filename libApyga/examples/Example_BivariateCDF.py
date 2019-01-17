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
import itertools as itt
import matplotlib as mpl
mpl.use("Qt5Agg") ## In macos, Qt5 works better for 3d plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import Apyga.stats as apyst


##########
## main ##
##########

if __name__ == "__main__":
	
	## Mixture of two normal laws and SparseHistogram
	X = np.zeros( (20000,2) )
	X[:10000,] = np.random.multivariate_normal( mean = [2,2] , cov = np.identity(2) , size = 10000 )
	X[10000:,] = np.random.multivariate_normal( mean = [-2,-2] , cov = [ [5,0] , [0,0.1] ] , size = 10000 )
	muX = apyst.SparseHist( X , [0.1,0.1] )
	
	## Grid to estimate the cdf
	gp =  np.arange( -8 , 8 , 0.1 )
	grid = np.array( [ np.array( [x,y] ) for x,y in itt.product( gp , gp ) ] )
	cdfGrid = muX.cdf(grid)
	pdfGrid = muX.pdf(grid)
	
	## Plot
	mx,my = np.meshgrid( gp , gp )
	cdf = cdfGrid.reshape( (gp.size,gp.size) )
	pdf = pdfGrid.reshape( (gp.size,gp.size) )

	fig = plt.figure( figsize = (14,7) )

	ax = fig.add_subplot(1,2,1,projection="3d")
	ax.plot_surface( mx , my , cdf , cmap = plt.cm.inferno , linewidth = 0 , antialiased = True )
	ax.set_xlabel( "x" )
	ax.set_ylabel( "y" )
	ax.set_zlabel( "cdf" )

	ax = fig.add_subplot(1,2,2,projection="3d")
	ax.plot_surface( mx , my , pdf , cmap = plt.cm.inferno , linewidth = 0 , antialiased = True )
	ax.set_xlabel( "x" )
	ax.set_ylabel( "y" )
	ax.set_zlabel( "pdf" )

	plt.tight_layout()
	plt.show()

	print("Done")


