
/********************************************************************************/
/********************************************************************************/
/*                                                                              */
/* Copyright Yoann Robin, 2018                                                  */
/*                                                                              */
/* yoann.robin.k@gmail.com                                                      */
/*                                                                              */
/* This software is a computer program that is part of the Apyga library. This  */
/* library makes it possible to study dynamic systems and to statistically      */
/* correct uni / multivariate data by applying optimal transport to             */
/* sparse histograms.                                                           */
/*                                                                              */
/* This software is governed by the CeCILL-C license under French law and       */
/* abiding by the rules of distribution of free software.  You can  use,        */
/* modify and/ or redistribute the software under the terms of the CeCILL-C     */
/* license as circulated by CEA, CNRS and INRIA at the following URL            */
/* "http://www.cecill.info".                                                    */
/*                                                                              */
/* As a counterpart to the access to the source code and  rights to copy,       */
/* modify and redistribute granted by the license, users are provided only      */
/* with a limited warranty  and the software's author,  the holder of the       */
/* economic rights,  and the successive licensors  have only  limited           */
/* liability.                                                                   */
/*                                                                              */
/* In this respect, the user's attention is drawn to the risks associated       */
/* with loading,  using,  modifying and/or developing or reproducing the        */
/* software by the user in light of its specific status of free software,       */
/* that may mean  that it is complicated to manipulate,  and  that  also        */
/* therefore means  that it is reserved for developers  and  experienced        */
/* professionals having in-depth computer knowledge. Users are therefore        */
/* encouraged to load and test the software's suitability as regards their      */
/* requirements in conditions enabling the security of their systems and/or     */
/* data to be ensured and,  more generally, to use and operate it in the        */
/* same conditions as regards security.                                         */
/*                                                                              */
/* The fact that you are presently reading this means that you have had         */
/* knowledge of the CeCILL-C license and that you accept its terms.             */
/*                                                                              */
/********************************************************************************/
/********************************************************************************/

/********************************************************************************/
/********************************************************************************/
/*                                                                              */
/* Copyright Yoann Robin, 2018                                                  */
/*                                                                              */
/* yoann.robin.k@gmail.com                                                      */
/*                                                                              */
/* Ce logiciel est un programme informatique faisant partie de la librairie     */
/* Apyga. Cette librairie permet d'étudier les systèmes dynamique et de         */
/* corriger statistiquement des données en uni/multivarié en appliquant le      */
/* transport optimal à des histogrammes creux.                                  */
/*                                                                              */
/* Ce logiciel est régi par la licence CeCILL-C soumise au droit français et    */
/* respectant les principes de diffusion des logiciels libres. Vous pouvez      */
/* utiliser, modifier et/ou redistribuer ce programme sous les conditions       */
/* de la licence CeCILL-C telle que diffusée par le CEA, le CNRS et l'INRIA     */
/* sur le site "http://www.cecill.info".                                        */
/*                                                                              */
/* En contrepartie de l'accessibilité au code source et des droits de copie,    */
/* de modification et de redistribution accordés par cette licence, il n'est    */
/* offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,    */
/* seule une responsabilité restreinte pèse sur l'auteur du programme, le       */
/* titulaire des droits patrimoniaux et les concédants successifs.              */
/*                                                                              */
/* A cet égard  l'attention de l'utilisateur est attirée sur les risques        */
/* associés au chargement,  à l'utilisation,  à la modification et/ou au        */
/* développement et à la reproduction du logiciel par l'utilisateur étant       */
/* donné sa spécificité de logiciel libre, qui peut le rendre complexe à        */
/* manipuler et qui le réserve donc à des développeurs et des professionnels    */
/* avertis possédant  des  connaissances  informatiques approfondies.  Les      */
/* utilisateurs sont donc invités à charger  et  tester  l'adéquation  du       */
/* logiciel à leurs besoins dans des conditions permettant d'assurer la         */
/* sécurité de leurs systèmes et ou de leurs données et, plus généralement,     */
/* à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.           */
/*                                                                              */
/* Le fait que vous puissiez accéder à cet en-tête signifie que vous avez       */
/* pris connaissance de la licence CeCILL-C, et que vous en avez accepté les    */
/* termes.                                                                      */
/*                                                                              */
/********************************************************************************/
/********************************************************************************/

#ifndef ACYGA_STATS_SPARSEHISTHPP
#define ACYGA_STATS_SPARSEHISTHPP


//-----------//
// Libraries //
//-----------//

#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include <cmath>

namespace Acyga { namespace stats {

template <class VectValue,class VectIndex,class DataType>
class SparseHist 
{
	public:
	
	// Some typedef {{{
	typedef unsigned int size_type ;
	typedef std::function<bool(const VectIndex&,const VectIndex&)> OrderType ;
	typedef std::map<VectIndex,int,OrderType> HashTable ;
	
	//}}}
	
	// Constructor / Destructor {{{
	SparseHist():
		m_dim(0) ,
		m_size(0) ,
		m_bin_origin() ,
		m_bin_width() ,
		m_alpha() ,
		m_beta() ,
		m_map( []( const VectIndex& x , const VectIndex& y ) { return std::lexicographical_compare( x.begin() , x.end() , y.begin() , y.end() ) ; } ) ,
		m_c() ,
		m_p()
	{}

	SparseHist( const DataType& X , const VectValue& bin_width , const VectValue& bin_origin ):
		m_dim(bin_width.size()) ,
		m_size(0) ,
		m_bin_origin(bin_origin) ,
		m_bin_width(bin_width) ,
		m_alpha(m_dim) ,
		m_beta(m_dim) ,
		m_map( []( const VectIndex& x , const VectIndex& y ) { return std::lexicographical_compare( x.begin() , x.end() , y.begin() , y.end() ) ; } ) ,
		m_c() ,
		m_p()
	{
		// Paramètres
		for( size_type s = 0 ; s < m_dim ; ++s )
		{
			m_alpha[s] = 1. / m_bin_width[s] ;
			m_beta[s] = - m_bin_origin[s] * m_alpha[s] ;
		}
		
		// Estimation des bins
		for( auto& x : X )
		{
			m_map[ bin_index(x) ]++ ;
		}
		
		// Construction finale
		m_size = m_map.size() ;
		double dsize = static_cast<double>(X.size()) ;
		m_p.resize( m_size ) ;
		m_c.resize( m_size ) ;
		size_type s = 0 ;
		for( auto& keyval : m_map )
		{
			m_p[s] = keyval.second / dsize ;
			m_c[s++] = bin_center(keyval.first) ;
		}
	}

	virtual ~SparseHist()
	{}
	
	// }}}
	
	// Accessors {{{
	
	size_type dim()
	{ return m_dim ; }
	
	size_type size()
	{ return m_size ; }
	
	VectValue bin_width()
	{ return m_bin_width ; }

	VectValue bin_origin()
	{ return m_bin_origin ; }
	
	DataType& c()
	{ return std::ref(m_c) ; }

	VectValue& p()
	{ return std::ref(m_p) ; }

	//}}}

	// Methods {{{
	
	inline VectIndex bin_index( const VectValue& x )
	{
		VectIndex index( m_dim ) ;
		for( size_type s = 0 ; s < m_dim ; ++s )
		{
			index[s] = std::floor( m_alpha[s] * x[s] + m_beta[s] ) ;
		}
		return index ;
	}
	
	inline VectValue bin_center( const VectIndex& index )
	{
		VectValue x(m_dim) ;
		for( size_type s = 0 ; s < m_dim ; ++s )
		{
			x[s] = m_bin_origin[s] + m_bin_width[s] * static_cast<double>(index[s]) + m_bin_width[s] / 2. ;
		}
		return x ;
	}

	VectIndex argwhere( const DataType& X )
	{
		VectIndex index ;
		VectIndex lIndex(X.size()) ;
		typename HashTable::iterator it ;
		size_type s = 0 ;
		for( auto& x : X )
		{
			index = bin_index(x) ;
			it = m_map.find( index ) ;
			lIndex[s++] = static_cast<int>( ( it == m_map.end() ) ? -1 : std::distance( m_map.begin() , it ) ) ;
		}
		return lIndex ;
	}
	
	// }}}
	
	protected:
	// Arguments {{{

	size_type	m_dim ;
	size_type	m_size ;
	VectValue	m_bin_origin ;
	VectValue	m_bin_width ;
	VectValue	m_alpha ;
	VectValue	m_beta ;
	HashTable	m_map ;
	DataType		m_c ;
	VectValue	m_p ;
	
	//}}}
} ; // class SparseHist 


}} // namespace Acyga::stats

#endif
