/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013-2022 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */



#include "tools/Communicator.h"
#include "tools/OpenMP.h"
#include "tools/SwitchingFunction.h"
#include "colvar/ActionRegister.h"
#include "colvar/Colvar.h"
#include "tools/Vector.h"
#include "tools/Pbc.h"
#include "tools/AtomNumber.h"
#include "tools/Tools.h"


#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>



namespace PLMD {

class Pbc;
class Communicator;

/// \ingroup TOOLBOX

/// A class that implements neighbor lists from two lists or a single list of atoms
class NeighborList
{
  bool reduced;
  bool serial_;
  bool do_pair_,do_pbc_,twolists_;
  const PLMD::Pbc* pbc_;
  Communicator& comm;
  std::vector<PLMD::AtomNumber> fullatomlist_,requestlist_;
  std::vector<std::pair<unsigned,unsigned> > neighbors_;
  double distance_;
  unsigned stride_,nlist0_,nlist1_,nallpairs_,lastupdate_;
  double zshift_;
  double xyradius_;   
/// Initialize the neighbor list with all possible pairs
  void initialize();
/// Return the pair of indexes in the positions array
/// of the two atoms forming the i-th pair among all possible pairs
  std::pair<unsigned,unsigned> getIndexPair(unsigned i);
/// Extract the list of atoms from the current list of close pairs
  void setRequestList();
public:
  NeighborList(const std::vector<PLMD::AtomNumber>& list0,
               const std::vector<PLMD::AtomNumber>& list1,
               const bool& serial,
               const bool& do_pair, const bool& do_pbc, const PLMD::Pbc& pbc, Communicator &cm,
               const double& distance=1.0e+30, const unsigned& stride=0,
               const double& zshift=0,
               const double& xyradius=0);
  NeighborList(const std::vector<PLMD::AtomNumber>& list0,
               const bool& serial,
               const bool& do_pbc,
               const PLMD::Pbc& pbc, Communicator &cm, const double& distance=1.0e+30,
               const unsigned& stride=0,
               const double& zshift=0,
               const double& xyradius=0);

/// Return the list of all atoms. These are needed to rebuild the neighbor list.
  std::vector<PLMD::AtomNumber>& getFullAtomList();
/// Update the indexes in the neighbor list to match the
/// ordering in the new positions array
/// and return the new list of atoms that must be requested to the main code
  std::vector<PLMD::AtomNumber>& getReducedAtomList();
/// Update the neighbor list and prepare the new
/// list of atoms that will be requested to the main code
  void update(const std::vector<PLMD::Vector>& positions);
/// Get the update stride of the neighbor list
  unsigned getStride() const;
/// Get the last step in which the neighbor list was updated
  unsigned getLastUpdate() const;
/// Set the step of the last update
  void setLastUpdate(unsigned step);
/// Get the size of the neighbor list
  unsigned size() const;
/// Get the i-th pair of the neighbor list
  std::pair<unsigned,unsigned> getClosePair(unsigned i) const;
/// Get the list of neighbors of the i-th atom
  std::vector<unsigned> getNeighbors(unsigned i);
  ~NeighborList() {}
/// Get the i-th pair of AtomNumbers from the neighbor list
  std::pair<AtomNumber,AtomNumber> getClosePairAtomNumber(unsigned i) const;
};

}





namespace PLMD {

NeighborList::NeighborList(const std::vector<AtomNumber>& list0, const std::vector<AtomNumber>& list1,
                           const bool& serial, const bool& do_pair, const bool& do_pbc, const Pbc& pbc, Communicator& cm,
                           const double& distance, const unsigned& stride, const double& zshift, const double& xyradius): reduced(false),
  serial_(serial), do_pair_(do_pair), do_pbc_(do_pbc), pbc_(&pbc), comm(cm),
  distance_(distance), stride_(stride), zshift_(zshift), xyradius_(xyradius)
{
// store full list of atoms needed
  fullatomlist_=list0;
  fullatomlist_.insert(fullatomlist_.end(),list1.begin(),list1.end());
  nlist0_=list0.size();
  nlist1_=list1.size();
  twolists_= true;
  if(!do_pair) {
    nallpairs_=nlist0_*nlist1_;
  } else {
    plumed_assert(nlist0_==nlist1_) << "when using PAIR option, the two groups should have the same number of elements\n"
                                    << "the groups you specified have size "<<nlist0_<<" and "<<nlist1_;
    nallpairs_=nlist0_;
  }
  initialize();
  lastupdate_=0;
}

NeighborList::NeighborList(const std::vector<AtomNumber>& list0, const bool& serial, const bool& do_pbc,
                           const Pbc& pbc, Communicator& cm, const double& distance,
                           const unsigned& stride, const double& zshift, const double& xyradius): reduced(false),
  serial_(serial), do_pbc_(do_pbc), pbc_(&pbc), comm(cm),
  distance_(distance), stride_(stride),  zshift_(zshift), xyradius_(xyradius) {
  fullatomlist_=list0;
  nlist0_=list0.size();
  twolists_=false;
  nallpairs_=nlist0_*(nlist0_-1)/2;
  initialize();
  lastupdate_=0;
}

void NeighborList::initialize() {
  neighbors_.clear();
  for(unsigned int i=0; i<nallpairs_; ++i) {
    neighbors_.push_back(getIndexPair(i));
  }
}

std::vector<AtomNumber>& NeighborList::getFullAtomList() {
  return fullatomlist_;
}

std::pair<unsigned,unsigned> NeighborList::getIndexPair(unsigned ipair) {
  std::pair<unsigned,unsigned> index;
  if(twolists_ && do_pair_) {
    index=std::pair<unsigned,unsigned>(ipair,ipair+nlist0_);
  } else if (twolists_ && !do_pair_) {
    index=std::pair<unsigned,unsigned>(ipair/nlist1_,ipair%nlist1_+nlist0_);
  } else if (!twolists_) {
    unsigned ii = nallpairs_-1-ipair;
    unsigned  K = unsigned(std::floor((std::sqrt(double(8*ii+1))+1)/2));
    unsigned jj = ii-K*(K-1)/2;
    index=std::pair<unsigned,unsigned>(nlist0_-1-K,nlist0_-1-jj);
  }
  return index;
}

void NeighborList::update(const std::vector<Vector>& positions) {
  neighbors_.clear();
  const double d2=distance_*distance_;
  // check if positions array has the correct length
  plumed_assert(positions.size()==fullatomlist_.size());

  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  unsigned nt=OpenMP::getNumThreads();
  if(serial_) {
    stride=1;
    rank=0;
    nt=1;
  }
  std::vector<unsigned> local_flat_nl;

  #pragma omp parallel num_threads(nt)
  {
    std::vector<unsigned> private_flat_nl;
    #pragma omp for nowait
    for(unsigned int i=rank; i<nallpairs_; i+=stride) {
      std::pair<unsigned,unsigned> index=getIndexPair(i);
      unsigned index0=index.first;
      unsigned index1=index.second;
      if(positions[index1][2] < positions[index0][2] + zshift_) continue;
      Vector distance;
      if(do_pbc_) {
        distance=pbc_->distance(positions[index0],positions[index1]);
      } else {
        distance=delta(positions[index0],positions[index1]);
      }

      if(xyradius_>0.0) {
      if(pow(distance[0],2) + pow(distance[1],2) > pow(xyradius_,2)) continue;
      }

      double value=modulo2(distance);

      if(value<=d2) {
        private_flat_nl.push_back(index0);
        private_flat_nl.push_back(index1);
      }
    }
    #pragma omp critical
    local_flat_nl.insert(local_flat_nl.end(), private_flat_nl.begin(), private_flat_nl.end());
  }

  // find total dimension of neighborlist
  std::vector <int> local_nl_size(stride, 0);
  local_nl_size[rank] = local_flat_nl.size();
  if(!serial_) comm.Sum(&local_nl_size[0], stride);
  int tot_size = std::accumulate(local_nl_size.begin(), local_nl_size.end(), 0);
  if(tot_size==0) {setRequestList(); return;}
  // merge
  std::vector<unsigned> merge_nl(tot_size, 0);
  // calculate vector of displacement
  std::vector<int> disp(stride);
  disp[0] = 0;
  int rank_size = 0;
  for(unsigned i=0; i<stride-1; ++i) {
    rank_size += local_nl_size[i];
    disp[i+1] = rank_size;
  }
  // Allgather neighbor list
  if(comm.initialized()&&!serial_) comm.Allgatherv((!local_flat_nl.empty()?&local_flat_nl[0]:NULL), local_nl_size[rank], &merge_nl[0], &local_nl_size[0], &disp[0]);
  else merge_nl = local_flat_nl;
  // resize neighbor stuff
  neighbors_.resize(tot_size/2);
  for(unsigned int i=0; i<tot_size/2; i++) {
    unsigned j=2*i;
    neighbors_[i] = std::make_pair(merge_nl[j],merge_nl[j+1]);
  }

  setRequestList();
}

void NeighborList::setRequestList() {
  requestlist_.clear();
  for(unsigned int i=0; i<size(); ++i) {
    requestlist_.push_back(fullatomlist_[neighbors_[i].first]);
    requestlist_.push_back(fullatomlist_[neighbors_[i].second]);
  }
  Tools::removeDuplicates(requestlist_);
  reduced=false;
}

std::vector<AtomNumber>& NeighborList::getReducedAtomList() {
  if(!reduced)for(unsigned int i=0; i<size(); ++i) {
      unsigned newindex0=0,newindex1=0;
      AtomNumber index0=fullatomlist_[neighbors_[i].first];
      AtomNumber index1=fullatomlist_[neighbors_[i].second];
// I exploit the fact that requestlist_ is an ordered vector
      auto p = std::find(requestlist_.begin(), requestlist_.end(), index0); plumed_dbg_assert(p!=requestlist_.end()); newindex0=p-requestlist_.begin();
      p = std::find(requestlist_.begin(), requestlist_.end(), index1); plumed_dbg_assert(p!=requestlist_.end()); newindex1=p-requestlist_.begin();
      neighbors_[i]=std::pair<unsigned,unsigned>(newindex0,newindex1);
    }
  reduced=true;
  return requestlist_;
}

unsigned NeighborList::getStride() const {
  return stride_;
}

unsigned NeighborList::getLastUpdate() const {
  return lastupdate_;
}

void NeighborList::setLastUpdate(unsigned step) {
  lastupdate_=step;
}

unsigned NeighborList::size() const {
  return neighbors_.size();
}

std::pair<unsigned,unsigned> NeighborList::getClosePair(unsigned i) const {
  return neighbors_[i];
}

std::pair<AtomNumber,AtomNumber> NeighborList::getClosePairAtomNumber(unsigned i) const {
  std::pair<AtomNumber,AtomNumber> Aneigh;
  Aneigh=std::pair<AtomNumber,AtomNumber>(fullatomlist_[neighbors_[i].first],fullatomlist_[neighbors_[i].second]);
  return Aneigh;
}

std::vector<unsigned> NeighborList::getNeighbors(unsigned index) {
  std::vector<unsigned> neighbors;
  for(unsigned int i=0; i<size(); ++i) {
    if(neighbors_[i].first==index)  neighbors.push_back(neighbors_[i].second);
    if(neighbors_[i].second==index) neighbors.push_back(neighbors_[i].first);
  }
  return neighbors;
}

}



// base coordination class
namespace PLMD {

class NeighborList;

namespace colvar {

class CoordinationBase : public Colvar {
  bool pbc;
  bool serial;
  std::unique_ptr<NeighborList> nl;
  bool invalidateList;
  bool firsttime;
  float z_shift;
  float xy_radius;

public:
  explicit CoordinationBase(const ActionOptions&);
  ~CoordinationBase();
// active methods:
  void calculate() override;
  void prepare() override;
  virtual double pairing(double distance,double&dfunc,unsigned i,unsigned j,int component)const=0;
  static void registerKeywords( Keywords& keys );
  int n_out;

};

}
}


// edited coordination class
namespace PLMD {    
namespace colvar {
class CoordinationMulti : public CoordinationBase {
//   SwitchingFunction switchingFunction;
  std::vector<SwitchingFunction> switching_functions;

public:
  explicit CoordinationMulti(const ActionOptions&);
// active methods:
  static void registerKeywords( Keywords& keys );
  double pairing(double distance,double&dfunc,unsigned i,unsigned j,int component)const override;
};

PLUMED_REGISTER_ACTION(CoordinationMulti,"COORDINATION_MULTI")

void CoordinationMulti::registerKeywords( Keywords& keys ) {
  CoordinationBase::registerKeywords(keys);
  keys.add("compulsory","NN","6","The n parameter of the switching function ");
  keys.add("compulsory","MM","0","The m parameter of the switching function; 0 implies 2*NN");
  keys.add("compulsory","D_0","0.0","The d_0 parameter of the switching function");
  keys.add("compulsory","R_0","The r_0 parameter of the switching function");
  keys.add("compulsory","D_MAX","The d_max parameter of the switching function");
  keys.addOutputComponent("0", "default", "Model outputs");
  keys.addOutputComponent("1", "default", "Model outputs");
  keys.addOutputComponent("2", "default", "Model outputs");
  keys.addOutputComponent("3", "default", "Model outputs");
  keys.addOutputComponent("4", "default", "Model outputs");
  keys.addOutputComponent("5", "default", "Model outputs");
  keys.addOutputComponent("6", "default", "Model outputs");
  keys.addOutputComponent("7", "default", "Model outputs");
  keys.addOutputComponent("8", "default", "Model outputs");
  keys.addOutputComponent("9", "default", "Model outputs");
  keys.addOutputComponent("10", "default", "Model outputs");
  keys.addOutputComponent("11", "default", "Model outputs");
  keys.addOutputComponent("12", "default", "Model outputs");
  keys.addOutputComponent("13", "default", "Model outputs");
  keys.addOutputComponent("14", "default", "Model outputs");
  keys.addOutputComponent("15", "default", "Model outputs");
  keys.addOutputComponent("16", "default", "Model outputs");
  keys.addOutputComponent("17", "default", "Model outputs");
  keys.addOutputComponent("18", "default", "Model outputs");
  keys.addOutputComponent("19", "default", "Model outputs");
  keys.addOutputComponent("20", "default", "Model outputs");
  keys.addOutputComponent("21", "default", "Model outputs");
  keys.addOutputComponent("22", "default", "Model outputs");
  keys.addOutputComponent("23", "default", "Model outputs");
  keys.addOutputComponent("24", "default", "Model outputs");
  keys.addOutputComponent("25", "default", "Model outputs");
  keys.addOutputComponent("26", "default", "Model outputs");
  keys.addOutputComponent("27", "default", "Model outputs");
  keys.addOutputComponent("28", "default", "Model outputs");
  keys.addOutputComponent("29", "default", "Model outputs");  
  keys.addOutputComponent("30", "default", "Model outputs");
  keys.addOutputComponent("31", "default", "Model outputs");
  keys.addOutputComponent("32", "default", "Model outputs");
  keys.addOutputComponent("33", "default", "Model outputs");
  keys.addOutputComponent("34", "default", "Model outputs");
  keys.addOutputComponent("35", "default", "Model outputs");
  keys.addOutputComponent("36", "default", "Model outputs");
  keys.addOutputComponent("37", "default", "Model outputs");
  keys.addOutputComponent("38", "default", "Model outputs");
  keys.addOutputComponent("39", "default", "Model outputs");
  keys.addOutputComponent("40", "default", "Model outputs");
}

CoordinationMulti::CoordinationMulti(const ActionOptions&ao):
  Action(ao),
  CoordinationBase(ao)
{
  int nn=6;
  int mm=0;
  double d0=0.0;
  double dmax=0.0;
  double r0=0.0;
  std::vector<float> cutoffs;
  parseVector("R_0", cutoffs);
  parse("D_0",d0);
  parse("D_MAX",dmax);
  parse("NN",nn);
  parse("MM",mm);

  if((int)cutoffs.size() != n_out)
  error("Number of R_0 values (" + std::to_string(cutoffs.size()) +
        ") must match N_OUT (" + std::to_string(n_out) + ")");
   
  // initialize n_out switching functions with different cutoffs 
  for(int i=0; i<n_out; i++){
    r0 = cutoffs[i];

    // string for initialization, otherwise no dmax 
    std::ostringstream oss;
    std::string sw, errors;
    oss << "RATIONAL D_0=" << d0 << " R_0=" << r0 << " NN=" << nn << " MM=" << mm << " D_MAX=" << dmax;
    sw = oss.str();

    // intialize switch and add to list
    SwitchingFunction aux;
    aux.set(sw,errors);
    switching_functions.push_back(aux);
    }

    // log recap
    for(int i=0; i<n_out; i++){
        log<<"switching function " << i <<": contacts are counted with cutoff "<<switching_functions[i].description()<<"\n";
    }
  checkRead();

}

// this selects the rigth switching function and applies it
double CoordinationMulti::pairing(double distance,double&dfunc,unsigned i,unsigned j,int component)const {
  (void) i; // avoid warnings
  (void) j; // avoid warnings
  return switching_functions[component].calculateSqr(distance,dfunc);
}

}

}



namespace PLMD {
namespace colvar {

void CoordinationBase::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbor list to speed up the calculation");
  keys.add("compulsory","N_OUT","Number of coordination to be computed");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbor list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbor list");
  keys.add("optional","Z_SHIFT","Shift of z coordinate");
  keys.add("optional","XY_RADIUS","Cutoff radius on xy plane");
  keys.add("atoms","GROUPA","First list of atoms");
  keys.add("atoms","GROUPB","Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are counted)");
}

CoordinationBase::CoordinationBase(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false),
  invalidateList(true),
  firsttime(true)
{

  parseFlag("SERIAL",serial);

  std::vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  parseAtomList("GROUPB",gb_lista);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

  parse("N_OUT", n_out);

// neighbor list stuff
  bool doneigh=false;
  double nl_cut=0.0;
  int nl_st=0;
  parseFlag("NLIST",doneigh);
  if(doneigh) {
    parse("NL_CUTOFF",nl_cut);
    if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
    parse("NL_STRIDE",nl_st);
    if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

 parse("Z_SHIFT", z_shift);
 parse("XY_RADIUS", xy_radius);
  

for(int i=0; i<n_out; i++){
    std::string name_comp = std::to_string(i);
    addComponentWithDerivatives(name_comp); componentIsNotPeriodic(name_comp);
}


  if(gb_lista.size()>0) {
    if(doneigh)  nl=Tools::make_unique<NeighborList>(ga_lista,gb_lista,serial,false,pbc,getPbc(),comm,nl_cut,nl_st,z_shift, xy_radius);
    else         nl=Tools::make_unique<NeighborList>(ga_lista,gb_lista,serial,false,pbc,getPbc(),comm);
  } else {
    if(doneigh)  nl=Tools::make_unique<NeighborList>(ga_lista,serial,pbc,getPbc(),comm,nl_cut,nl_st,z_shift, xy_radius);
    else         nl=Tools::make_unique<NeighborList>(ga_lista,serial,pbc,getPbc(),comm);
  }

  requestAtoms(nl->getFullAtomList());

  log.printf("  between two groups of %u and %u atoms\n",static_cast<unsigned>(ga_lista.size()),static_cast<unsigned>(gb_lista.size()));
  log.printf("  first group:\n");
  for(unsigned int i=0; i<ga_lista.size(); ++i) {
    if ( (i+1) % 25 == 0 ) log.printf("  \n");
    log.printf("  %d", ga_lista[i].serial());
  }
  log.printf("  \n  second group:\n");
  for(unsigned int i=0; i<gb_lista.size(); ++i) {
    if ( (i+1) % 25 == 0 ) log.printf("  \n");
    log.printf("  %d", gb_lista[i].serial());
  }
  log.printf("  \n");
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(doneigh) {
    log.printf("  using neighbor lists with\n");
    log.printf("  update every %d steps and cutoff %f\n",nl_st,nl_cut);
  }
}

CoordinationBase::~CoordinationBase() {
// destructor required to delete forward declared class
}

void CoordinationBase::prepare() {
  if(nl->getStride()>0) {
    if(firsttime || (getStep()%nl->getStride()==0)) {
      requestAtoms(nl->getFullAtomList());
      invalidateList=true;
      firsttime=false;
    } else {
      requestAtoms(nl->getReducedAtomList());
      invalidateList=false;
      if(getExchangeStep()) error("Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!");
    }
    if(getExchangeStep()) firsttime=true;
  }
}

// calculator
void CoordinationBase::calculate()
{
if(nl->getStride()>0 && invalidateList) {
    nl->update(getPositions());
  }

  for(int component=0; component<n_out; component++){
  double ncoord=0.;
  Tensor virial;
  std::vector<Vector> deriv(getNumberOfAtoms());

  unsigned stride;
  unsigned rank;
  if(serial) {
    stride=1;
    rank=0;
  } else {
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }

  unsigned nt=OpenMP::getNumThreads();
  const unsigned nn=nl->size();
  if(nt*stride*10>nn) nt=1;

  #pragma omp parallel num_threads(nt)
  {
    std::vector<Vector> omp_deriv(getPositions().size());
    Tensor omp_virial;

    #pragma omp for reduction(+:ncoord) nowait
    for(unsigned int i=rank; i<nn; i+=stride) {

      Vector distance;
      unsigned i0=nl->getClosePair(i).first;
      unsigned i1=nl->getClosePair(i).second;

      if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;

      if(pbc) {
        distance=pbcDistance(getPosition(i0),getPosition(i1));
      } else {
        distance=delta(getPosition(i0),getPosition(i1));
      }

      double dfunc=0.;
      ncoord += pairing(distance.modulo2(), dfunc,i0,i1, component);

      Vector dd(dfunc*distance);
      Tensor vv(dd,distance);
      if(nt>1) {
        omp_deriv[i0]-=dd;
        omp_deriv[i1]+=dd;
        omp_virial-=vv;
      } else {
        deriv[i0]-=dd;
        deriv[i1]+=dd;
        virial-=vv;
      }

    }
    #pragma omp critical
    if(nt>1) {
      for(unsigned i=0; i<getPositions().size(); i++) deriv[i]+=omp_deriv[i];
      virial+=omp_virial;
    }
  }

  if(!serial) {
    comm.Sum(ncoord);
    if(!deriv.empty()) comm.Sum(&deriv[0][0],3*deriv.size());
    comm.Sum(virial);
  }

  std::string name_comp = std::to_string(component);
  Value* value=getPntrToComponent(name_comp);
  for(unsigned i=0; i<deriv.size(); ++i) setAtomsDerivatives(value,i,deriv[i]);
  value->set(ncoord);
  setBoxDerivatives  (value, virial);
  }
}
}
}
