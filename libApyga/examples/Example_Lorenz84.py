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
import matplotlib as mpl
mpl.use("Qt5Agg") ## In macos, Qt5 works better for 3d plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import Apyga.dynamic.continuous as apydc


##########
## main ##
##########

if __name__ == "__main__":
	
	## Construction of Lorenz84
	l84 = apydc.Lorenz84( size = 1000 , F = "cyclic" )
	
	## Building 1000 orbites, i.e. a sequence of snapshot
	time = np.arange( 0 , 7 * 73 , 0.005 )
	orbit = l84.orbit( time )
	
	## Extract seasons index
	lenYear = int( len(time) / 7 )
	iFa0 = 5 * lenYear
	iWi0 = 5 * lenYear + int(lenYear/4)
	iSp0 = 5 * lenYear + int(lenYear/2)
	iSu0 = 5 * lenYear + 3 * int(lenYear/4)
	iFa1 = 6 * lenYear
	iWi1 = 6 * lenYear + int(lenYear/4)
	iSp1 = 6 * lenYear + int(lenYear/2)
	iSu1 = 6 * lenYear + 3 * int(lenYear/4)
	

	## Plot orbits

	fig = plt.figure( figsize = (16,8) )
	xLim = (-1,3)
	yLim = (-2.5,2.5)
	zLim = (-2.5,2.5)
	title = [ "Fall" , "Winter" , "Spring" , "Summer" ]

	for i,seas in enumerate([iFa0,iWi0,iSp0,iSu0]):
		ax = fig.add_subplot( 2 , 4  , i + 1 , projection = "3d" )
		ax.plot( orbit[seas,:,0] , orbit[seas,:,1] , orbit[seas,:,2] , color = "blue" , linestyle = "" , marker = "o" )
		ax.set_title( title[i] )
		ax.set_xlabel( "x" )
		ax.set_ylabel( "y" )
		ax.set_zlabel( "z" )
		ax.set_xlim( xLim )
		ax.set_ylim( yLim )
		ax.set_zlim( zLim )
		ax.set_xticks( [-1,0,1,2,3] )
		ax.set_yticks( [-2,-1,0,1,2] )
		ax.set_zticks( [-2,-1,0,1,2] )
	
	for i,seas in enumerate([iFa1,iWi1,iSp1,iSu1]):
		ax = fig.add_subplot( 2 , 4  , i + 4 + 1 , projection = "3d" )
		ax.plot( orbit[seas,:,0] , orbit[seas,:,1] , orbit[seas,:,2] , color = "blue" , linestyle = "" , marker = "o" )
		ax.set_xlabel( "x" )
		ax.set_ylabel( "y" )
		ax.set_zlabel( "z" )
		ax.set_xlim( xLim )
		ax.set_ylim( yLim )
		ax.set_zlim( zLim )
		ax.set_xticks( [-1,0,1,2,3] )
		ax.set_yticks( [-2,-1,0,1,2] )
		ax.set_zticks( [-2,-1,0,1,2] )
	
	plt.tight_layout()
	plt.show()

	print("Done")
