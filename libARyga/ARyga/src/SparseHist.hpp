
/********************************************************************************/
/********************************************************************************/
/*                                                                              */
/* Copyright Yoann Robin, 2018                                                  */
/*                                                                              */
/* yoann.robin.k@gmail.com                                                      */
/*                                                                              */
/* This software is a computer program that is part of the ARyga library. This   */
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
/* ARyga. Cette librairie permet d'étudier les systèmes dynamique et de          */
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


#ifndef ARYGA_SPARSEHIST_INCLUDED
#define ARYGA_SPARSEHIST_INCLUDED

Rcpp::NumericVector binwidth_estimator( Rcpp::NumericMatrix X , std::string method = "auto" )
{
	int nrow = X.nrow() ;
	int ncol = X.ncol() ;
	
	if( method == "auto" )
	{
		method = (ncol < 1000) ? "Sturges" : "FD" ;
	}
	
	Rcpp::NumericMatrix bin_width(nrow) ;
	
	if( method == "Sturges" )
	{
		double nh = std::log2(ncol) + 1. ;
		std::fill( bin_width.begin() , bin_width.end() , 1. / nh ) ;
	}
	else // FD (Freedman Diaconis) method, robust over outliners
	{
		double power = std::pow( ncol , 1./3. ) ;
		int q25 = static_cast<int>( ncol * 0.25 ) ;
		int q75 = static_cast<int>( ncol * 0.75 ) ;
		Rcpp::NumericVector Tmp(ncol) ;
		for( std::size_t i = 0 ; i < nrow ; ++i )
		{
			for( std::size_t j = 0 ; j < ncol ; ++j )
			{
				Tmp[j] = X(i,j) ;
			}
			std::sort( Tmp.begin() , Tmp.end() ) ;
			bin_width[i] = 2. * ( Tmp[q75] - Tmp[q25] ) / power ;
		}
	}
	return(bin_width) ;
}



//' SparseHistClass
//' 
//' @export SparseHistClass
class SparseHistClass
{
	public:
	//---------//
	// Typedef //
	//---------//
	
	typedef std::map<std::vector<int>,int,std::function<bool(const std::vector<int>&,const std::vector<int>&)>> HashTable ;
	
	//--------------//
	// Constructeur //
	//--------------//
	
	SparseHistClass():
		_dim() ,
		_bin_origin() ,
		_bin_width() ,
		_alpha() ,
		_beta() ,
		_map( []( const std::vector<int>& x , const std::vector<int>& y ) { return std::lexicographical_compare( x.begin() , x.end() , y.begin() , y.end() ) ; } ) ,
		_p() ,
		_c()
	{}

	SparseHistClass( Rcpp::NumericMatrix data , Rcpp::NumericVector bin_width , Rcpp::NumericVector bin_origin ):
		_dim(bin_width.size()) ,
		_bin_origin(bin_origin) ,
		_bin_width(bin_width) ,
		_alpha(_dim) ,
		_beta(_dim) ,
		_map( []( const std::vector<int>& x , const std::vector<int>& y ) { return std::lexicographical_compare( x.begin() , x.end() , y.begin() , y.end() ) ; } ) ,
		_p() ,
		_c()
	{
		// Paramètres de hash
		for( std::size_t s = 0 ; s < _bin_origin.size() ; ++s )
		{
			_alpha[s] = 1. / _bin_width[s] ;
			_beta[s] = - _bin_origin[s] * _alpha[s] ;
		}
		
		// Hashage des données (construction de l'histogramme!!)
		int ncol = data.ncol() ;
		for( int i = 0 ; i < ncol ; ++i )
		{
			_map[hash( data( Rcpp::_ , i ) )]++ ;
		}
		
		// Construction de p et c
		std::size_t _sizeMap = _map.size() ;
		double _sizeData = static_cast<double>(ncol) ;
		_p = Rcpp::NumericVector(_sizeMap) ;
		_c = Rcpp::NumericMatrix(_dim,_sizeMap) ;
		int i = 0 ;
		for( auto& x : _map )
		{
			_p[i] = static_cast<double>(x.second) / _sizeData ;
			_c(Rcpp::_,i) = hashi( x.first ) ;
			++i ;
		}
	}
	
	//------------//
	// Accesseurs //
	//------------//
	
	int size()
	{ return _map.size() ; }
	
	int dim()
	{ return _dim ; }
	
	Rcpp::NumericVector bin_width()
	{ return _bin_width ; }

	Rcpp::NumericVector bin_origin()
	{ return _bin_origin ; }
	
	Rcpp::NumericVector p()
	{ return _p ; }

	Rcpp::NumericMatrix c()
	{ return _c ; }

	//-------------------//
	// Méthodes internes //
	//-------------------//
	
	std::vector<int> hash( Rcpp::NumericMatrix::Column x )
	{
		std::vector<int> index(_dim) ;
		for( std::size_t s = 0 ; s < _dim ; ++s )
		{
			index[s] = std::floor( _alpha[s] * x[s] + _beta[s] ) ;
		}
		return index ;
	}
	
	Rcpp::NumericVector hashi( const std::vector<int>& index )
	{
		Rcpp::NumericVector val(_dim) ;
		for( std::size_t s = 0 ; s < _dim ; ++s )
		{
			val[s] = _bin_origin[s] + _bin_width[s] * static_cast<double>(index[s]) + _bin_width[s] / 2. ;
		}
		return val ;
	}
	
	//-----------------//
	// "Vrai" méthodes //
	//-----------------//
	
	Rcpp::NumericVector argwhere( Rcpp::NumericMatrix data )
	{
		std::size_t _sizeData(data.ncol()) ;
		Rcpp::NumericVector index(_sizeData) ;
		std::vector<int> hash_value ;
		std::size_t j = 0 ;
		HashTable::iterator it ;
		for( std::size_t s = 0 ; s < _sizeData ; ++s )
		{
			hash_value = hash( data(Rcpp::_,s) ) ;
			it = _map.find( hash_value ) ;
			index[j++] = static_cast<int>( (it == _map.end()) ? -1 : std::distance( _map.begin() , it ) + 1 ) ;
		}
		return index ;
	}
	
	
	//-----------//
	// Arguments //
	//-----------//
	private:
	int _dim ;
	Rcpp::NumericVector _bin_origin ;
	Rcpp::NumericVector _bin_width ;
	std::vector<double> _alpha ;
	std::vector<double> _beta ;
	HashTable _map ;
	Rcpp::NumericVector _p ;
	Rcpp::NumericMatrix _c ;
} ;

#endif
