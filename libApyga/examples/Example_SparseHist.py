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
import time
import Apyga.stats as apyst


#########################
## Test implementation ##
#########################

def single_test():

	## Parameters
	Time = np.zeros(8)
	dim = 40
	size = 10000
	low = np.array( [ -1e6 for i in range(dim) ] )
	high = np.array( [ 1e6 for i in range(dim) ] )
	bin_width = np.array( [ 0.1 for i in range(dim) ] )

	## Data Generation
	Time[0] = time.clock()
	X = np.random.uniform( low = low , high = high , size = (size,dim) )

	## Histogram generation
	Time[1] = time.clock()
	muX = apyst.SparseHist( X , bin_width )
	
	## Draw 10000 elements according to muX, and add an uniform noise around center of cells. Uniform law is draw in cell [center - noise ; center + noise]^dim
	Time[2] = time.clock()
	Y = muX.sample( size = 1000 , noise = 0.05 , law_noise = "uniform" )

	## Find Index of X
	Time[3] = time.clock()
	lIndX = muX.argwhere(X)

	## Find Index of Y
	Time[4] = time.clock()
	lIndY = muX.argwhere(Y)

	## Validity test for X
	Time[5] = time.clock()
	nErrorX = 0
	for i,ind in enumerate(lIndX):
		T = np.count_nonzero(X[i,:] < (muX.c[ind,:] - bin_width/2)) + np.count_nonzero(X[i,:] > (muX.c[ind,:] + bin_width/2))
		if T > 0:
			nErrorX += 1
	
	## Validity test for Y
	Time[6] = time.clock()
	nErrorY = 0
	for i,ind in enumerate(lIndY):
		T = np.count_nonzero(Y[i,:] < (muX.c[ind,:] - bin_width/2)) + np.count_nonzero(Y[i,:] > (muX.c[ind,:] + bin_width/2))
		if T > 0:
			nErrorY += 1
	
	## End
	Time[7] = time.clock()
	Diff = np.zeros(8)
	Diff[:7] = Time[1:] - Time[:7]
	Diff[7] = Time[-1] - Time[0]
	return Diff,nErrorX,nErrorY


def main_test():
	print("==========================================================")
	print("||         Apyga.stats.SparseHist benchmark test" )
	print("==========================================================")
	print("|| Parameters :")
	print("||    => X = 10000 elements in [-1e6,1e6]^40")
	print("||    => Build with an uniform law on [-1e6,1e6]^40")
	print("||    => size of cells : 0.1 in each dimension")
	print("==========================================================")
	print("|| Tests :")
	print("||    => Construction of 40-dim histogram of X")
	print("||    => Draw Y = 1000 noised elements according to hist")
	print("||    => Find the cell index of the X")
	print("||    => Find the cell index of the Y")
	print("||    => Check cell of X" )
	print("||    => Check cell of Y" )
	print("||    => Running 30 times")
	print("==========================================================")
	print("|| Difficulty :")
	print("||    => Histogram is very very very sparse")
	print("==========================================================")
	
	nTest = 30
	Time = np.zeros( (nTest,8) )
	nErrorX = 0
	nErrorY = 0
	for i in range(nTest):
		print("|| In progress... {}%          \r".format( np.round( 100*i/nTest , 2 ) ) , end = "" )
		Time[i,:],nErrorX,nErrorY = single_test()
	print("|| In progress... 100%                  ")
	mTime = np.round( np.mean( Time , axis = 0 ) , 5 )
	sTime = np.round( np.std( Time , axis = 0 ) , 5 )

	print("==========================================================")
	print("|| Mean results :")
	print("||    => X generation           : {} +- {} sec".format( mTime[0] , sTime[0] ) )
	print("||    => Build Hist of X        : {} +- {} sec".format( mTime[1] , sTime[1] ) )
	print("||    => Draw Y elements        : {} +- {} sec".format( mTime[2] , sTime[2] ) )
	print("||    => Index of X             : {} +- {} sec".format( mTime[3] , sTime[3] ) )
	print("||    => Index of Y             : {} +- {} sec".format( mTime[4] , sTime[4] ) )
	print("||    => Validity test of X     : {} +- {} sec".format( mTime[5] , sTime[5] ) )
	print("||    => Validity test of Y     : {} +- {} sec".format( mTime[6] , sTime[6] ) )
	print("||    => Number of X cell error : {}".format(nErrorX) )
	print("||    => Number of Y cell error : {}".format(nErrorY) )
	print("||    => Single total time      : {} +- {} sec".format( mTime[7] , sTime[7] ) )
	print("||    => Total time             : {} sec".format( round( np.sum(Time[:,7]) , 5 ) ) )
	print("==========================================================")


##########
## main ##
##########

if __name__ == "__main__":
	main_test()

	print("Done")
