
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

/*
 * This file has been adapted by Nicolas Bonneel (2013), 
 * from full_graph.h from LEMON, a generic C++ optimization library,
 * to make the other files independant from the rest of 
 * the original library.
 * 
 *
 **** Original file Copyright Notice :
 * Copyright (C) 2003-2010
 * Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
 * (Egervary Research Group on Combinatorial Optimization, EGRES).
 *
 * Permission to use, modify and distribute this software is granted
 * provided that this copyright notice appears in all copies. For
 * precise terms see the accompanying LICENSE file.
 *
 * This software is provided "AS IS" with no warranty of any kind,
 * express or implied, and with no claim as to its suitability for any
 * purpose.
 *
 */

#ifndef LEMON_CORE_H
#define LEMON_CORE_H

#include <vector>
#include <algorithm>


// Disable the following warnings when compiling with MSVC:
// C4250: 'class1' : inherits 'class2::member' via dominance
// C4355: 'this' : used in base member initializer list
// C4503: 'function' : decorated name length exceeded, name was truncated
// C4800: 'type' : forcing value to bool 'true' or 'false' (performance warning)
// C4996: 'function': was declared deprecated
#ifdef _MSC_VER
#pragma warning( disable : 4250 4355 4503 4800 4996 )
#endif

///\file
///\brief LEMON core utilities.
///
///This header file contains core utilities for LEMON.
///It is automatically included by all graph types, therefore it usually
///do not have to be included directly.

namespace lemon {

  /// \brief Dummy type to make it easier to create invalid iterators.
  ///
  /// Dummy type to make it easier to create invalid iterators.
  /// See \ref INVALID for the usage.
  struct Invalid {
  public:
    bool operator==(Invalid) { return true;  }
    bool operator!=(Invalid) { return false; }
    bool operator< (Invalid) { return false; }
  };

  /// \brief Invalid iterators.
  ///
  /// \ref Invalid is a global type that converts to each iterator
  /// in such a way that the value of the target iterator will be invalid.
#ifdef LEMON_ONLY_TEMPLATES
  const Invalid INVALID = Invalid();
#else
  extern const Invalid INVALID;
#endif

  /// \addtogroup gutils
  /// @{

  ///Create convenience typedefs for the digraph types and iterators

  ///This \c \#define creates convenient type definitions for the following
  ///types of \c Digraph: \c Node,  \c NodeIt, \c Arc, \c ArcIt, \c InArcIt,
  ///\c OutArcIt, \c BoolNodeMap, \c IntNodeMap, \c DoubleNodeMap,
  ///\c BoolArcMap, \c IntArcMap, \c DoubleArcMap.
  ///
  ///\note If the graph type is a dependent type, ie. the graph type depend
  ///on a template parameter, then use \c TEMPLATE_DIGRAPH_TYPEDEFS()
  ///macro.
#define DIGRAPH_TYPEDEFS(Digraph)                                       \
//  typedef Digraph::Node Node;                                           \
//  typedef Digraph::Arc Arc;                                             \


  ///Create convenience typedefs for the digraph types and iterators

  ///\see DIGRAPH_TYPEDEFS
  ///
  ///\note Use this macro, if the graph type is a dependent type,
  ///ie. the graph type depend on a template parameter.
#define TEMPLATE_DIGRAPH_TYPEDEFS(Digraph)                              \
  typedef typename Digraph::Node Node;                                  \
  typedef typename Digraph::Arc Arc;                                    \

 

} //namespace lemon

#endif
