
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
 * to implement a lightweight fully connected bipartite graph. A previous
 * version of this file is used as part of the Displacement Interpolation 
 * project, 
 * Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
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

#ifndef LEMON_FULL_BIPARTITE_GRAPH_H
#define LEMON_FULL_BIPARTITE_GRAPH_H

#include "core.hpp"

///\ingroup graphs
///\file
///\brief FullBipartiteDigraph and FullBipartiteGraph classes.


namespace lemon {


  class FullBipartiteDigraphBase {
  public:

    typedef FullBipartiteDigraphBase Digraph;

    //class Node;
	typedef int Node;
    //class Arc;
	typedef long long Arc;

  protected:

    int _node_num;
    long long _arc_num;
	
    FullBipartiteDigraphBase() {}

    void construct(int n1, int n2) { _node_num = n1+n2; _arc_num = n1 * n2; _n1=n1; _n2=n2;}

  public:

	int _n1, _n2;


    Node operator()(int ix) const { return Node(ix); }
    static int index(const Node& node) { return node; }

    Arc arc(const Node& s, const Node& t) const {
		if (s<_n1 && t>=_n1)
			return Arc(s * _n2 + (t-_n1) );
		else
			return Arc(-1);
    }

    int nodeNum() const { return _node_num; }
    long long arcNum() const { return _arc_num; }

    int maxNodeId() const { return _node_num - 1; }
    long long maxArcId() const { return _arc_num - 1; }

    Node source(Arc arc) const { return arc / _n2; }
    Node target(Arc arc) const { return (arc % _n2) + _n1; }

    static int id(Node node) { return node; }
    static long long id(Arc arc) { return arc; }

    static Node nodeFromId(int id) { return Node(id);}
    static Arc arcFromId(int id) { return Arc(id);}


    Arc findArc(Node s, Node t, Arc prev = -1) const {
      return prev == -1 ? arc(s, t) : -1;
    }

    void first(Node& node) const {
      node = _node_num - 1;
    }

    static void next(Node& node) {
      --node;
    }

    void first(Arc& arc) const {
      arc = _arc_num - 1;
    }

    static void next(Arc& arc) {
      --arc;
    }

    void firstOut(Arc& arc, const Node& node) const {
		if (node>=_n1)
			arc = -1;
		else
			arc = (node + 1) * _n2 - 1;
    }

    void nextOut(Arc& arc) const {
      if (arc % _n2 == 0) arc = 0;
      --arc;
    }

    void firstIn(Arc& arc, const Node& node) const {
		if (node<_n1)
			arc = -1;
		else
			arc = _arc_num + node - _node_num;
    }

    void nextIn(Arc& arc) const {
      arc -= _n2;
      if (arc < 0) arc = -1;
    }

  };

  /// \ingroup graphs
  ///
  /// \brief A directed full graph class.
  ///
  /// FullBipartiteDigraph is a simple and fast implmenetation of directed full
  /// (complete) graphs. It contains an arc from each node to each node
  /// (including a loop for each node), therefore the number of arcs
  /// is the square of the number of nodes.
  /// This class is completely static and it needs constant memory space.
  /// Thus you can neither add nor delete nodes or arcs, however
  /// the structure can be resized using resize().
  ///
  /// This type fully conforms to the \ref concepts::Digraph "Digraph concept".
  /// Most of its member functions and nested classes are documented
  /// only in the concept class.
  ///
  /// This class provides constant time counting for nodes and arcs.
  ///
  /// \note FullBipartiteDigraph and FullBipartiteGraph classes are very similar,
  /// but there are two differences. While this class conforms only
  /// to the \ref concepts::Digraph "Digraph" concept, FullBipartiteGraph
  /// conforms to the \ref concepts::Graph "Graph" concept,
  /// moreover FullBipartiteGraph does not contain a loop for each
  /// node as this class does.
  ///
  /// \sa FullBipartiteGraph
  class FullBipartiteDigraph : public FullBipartiteDigraphBase {
    typedef FullBipartiteDigraphBase Parent;

  public:

    /// \brief Default constructor.
    ///
    /// Default constructor. The number of nodes and arcs will be zero.
    FullBipartiteDigraph() { construct(0,0); }

    /// \brief Constructor
    ///
    /// Constructor.
    /// \param n The number of the nodes.
    FullBipartiteDigraph(int n1, int n2) { construct(n1, n2); }


    /// \brief Returns the node with the given index.
    ///
    /// Returns the node with the given index. Since this structure is
    /// completely static, the nodes can be indexed with integers from
    /// the range <tt>[0..nodeNum()-1]</tt>.
    /// The index of a node is the same as its ID.
    /// \sa index()
    Node operator()(int ix) const { return Parent::operator()(ix); }

    /// \brief Returns the index of the given node.
    ///
    /// Returns the index of the given node. Since this structure is
    /// completely static, the nodes can be indexed with integers from
    /// the range <tt>[0..nodeNum()-1]</tt>.
    /// The index of a node is the same as its ID.
    /// \sa operator()()
    static int index(const Node& node) { return Parent::index(node); }

    /// \brief Returns the arc connecting the given nodes.
    ///
    /// Returns the arc connecting the given nodes.
    /*Arc arc(Node u, Node v) const {
      return Parent::arc(u, v);
    }*/

    /// \brief Number of nodes.
    int nodeNum() const { return Parent::nodeNum(); }
    /// \brief Number of arcs.
    long long arcNum() const { return Parent::arcNum(); }
  };




} //namespace lemon


#endif //LEMON_FULL_GRAPH_H
