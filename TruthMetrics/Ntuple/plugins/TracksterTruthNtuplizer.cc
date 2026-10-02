// Writes, per event, everything the trackster-building metrics need, so that the clustering can be
// re-run and scored outside CMSSW (the ticltune python package):
//
//   layer clusters   energy, position, layer, side, detector, seed DetId (CLUEstering tie-break tag),
//                    the iteration mask, and the energy that no in-time truth particle accounts for
//   truth table      sparse (layer cluster, base particle, energy): the base particle's share of the
//                    layer cluster, rechit-energy weighted (see below)
//   rechits          every HGCAL rechit in a layer cluster (energy, position, layer, the layer cluster
//                    holding most of it) and the same truth table per rechit: E_rechit * E_sim(b) / E_sim(all)
//   base particles   the caloBoundary particles that own calorimeter hits, with their interaction
//                    (signal or which in-time pileup collision) and their origin
//   CMSSW tracksters the CLUEstering assignment per layer cluster and any trackster collections, for
//                    the closure test of the emulator
//
// Base particle of a truth particle p (baseLevel = "crossing", the default): its nearest ancestor-or-self that
// crossed into the calorimeter (a boundary checkpoint in the graph, not back-scattered). Every crossing counts,
// also below another one: an electron that crossed after radiating keeps its own deposits, and each brem photon or
// conversion leg that crossed on its own keeps its own. The graph's caloBoundary *level* is an antichain (only the
// outermost crossing of nested ones), which would give the electron everything; baseLevel = "caloBoundary" uses it.
// A particle with no such ancestor (rare: hits from below the boundary that no crossing particle owns) falls back to
// its nearest reconstructableFinalState ancestor-or-self, then to its root, and is flagged by `bp_kind`.
//
// Energy of base particle b in layer cluster l:
//   s(l, b) = sum over the cells c of l:  fraction_l(c) * E_rechit(c) * E_sim(b, c) / E_sim(all in-time, c)
// A cell with a rechit but no in-time sim energy (out-of-time pileup, noise) goes to the layer
// cluster's no-truth energy. Out-of-time energy in a cell that also has in-time sim energy cannot be
// told apart and is shared among the in-time particles, as in the TICL associators.

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "TTree.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  constexpr int kNone = -1;
  enum BaseKind : int { kCaloBoundary = 0, kReconstructableFinalState = 1, kRoot = 2 };

  // First parent through the first production vertex that has an incoming particle; -1 for a root.
  int firstParent(truth::Graph const& graph, uint32_t p) {
    for (auto v : graph.productionVertices(p)) {
      auto in = graph.incomingParticles(v);
      if (!in.empty())
        return static_cast<int>(in[0]);
    }
    return kNone;
  }
}  // namespace

class TracksterTruthNtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TracksterTruthNtuplizer(edm::ParameterSet const&);
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  struct TracksterBranches {
    std::vector<float> energy;
    std::vector<int> nLC;
    std::vector<int> lc;
    std::vector<float> mult;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  const edm::EDGetTokenT<std::vector<float>> maskToken_;
  const edm::InputTag assignmentTag_;
  edm::EDGetTokenT<std::vector<int32_t>> assignmentToken_;
  const std::vector<edm::InputTag> trackstersTags_;
  const std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> trackstersTokens_;
  const std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const bool crossing_;  // baseLevel == "crossing": every boundary crossing, not only the caloBoundary level
  hgcal::RecHitTools rhtools_;

  TTree* tree_ = nullptr;
  unsigned run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  // layer clusters
  std::vector<float> lcE_, lcX_, lcY_, lcZ_, lcMask_, lcNoTruthE_, lcRecE_;
  std::vector<int> lcLayer_, lcSide_, lcDet_, lcNHits_, lcAlgo_, lcMissingRecHits_;
  std::vector<unsigned> lcSeed_;
  // truth table
  std::vector<int> trLC_, trBP_;
  std::vector<float> trE_;
  // rechits in layer clusters, and their truth table
  std::vector<float> rhE_, rhX_, rhY_, rhZ_, trhE_;
  std::vector<int> rhLayer_, rhDet_, rhLC_, trhRH_, trhBP_;
  // base particles
  std::vector<int> bpId_, bpPdg_, bpSignal_, bpBX_, bpEvt_, bpKind_, bpOrigin_, bpOriginPdg_;
  std::vector<float> bpE_, bpEta_, bpPhi_, bpSimE_, bpOriginE_;
  // CMSSW reference
  std::vector<int> clueAssignment_;
  std::vector<TracksterBranches> tracksters_;
};

TracksterTruthNtuplizer::TracksterTruthNtuplizer(edm::ParameterSet const& cfg)
    : graphToken_(consumes(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes(cfg.getParameter<edm::InputTag>("hitIndex"))),
      layerClustersToken_(consumes(cfg.getParameter<edm::InputTag>("layerClusters"))),
      maskToken_(consumes(cfg.getParameter<edm::InputTag>("layerClusterMask"))),
      assignmentTag_(cfg.getParameter<edm::InputTag>("clueAssignment")),
      trackstersTags_(cfg.getParameter<std::vector<edm::InputTag>>("tracksters")),
      trackstersTokens_(edm::vector_transform(
          trackstersTags_, [this](edm::InputTag const& t) { return consumes<std::vector<ticl::Trackster>>(t); })),
      recHitTokens_(edm::vector_transform(cfg.getParameter<std::vector<edm::InputTag>>("recHits"),
                                          [this](edm::InputTag const& t) { return consumes<HGCRecHitCollection>(t); })),
      geomToken_(esConsumes()),
      crossing_(cfg.getParameter<std::string>("baseLevel") == "crossing") {
  if (!crossing_ && cfg.getParameter<std::string>("baseLevel") != "caloBoundary")
    throw cms::Exception("Configuration") << "baseLevel must be crossing or caloBoundary";
  usesResource(TFileService::kSharedResource);
  if (!assignmentTag_.label().empty())
    assignmentToken_ = consumes(assignmentTag_);
  tracksters_.resize(trackstersTags_.size());
}

void TracksterTruthNtuplizer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("events", "trackster-building truth ntuple");
  tree_->Branch("run", &run_);
  tree_->Branch("lumi", &lumi_);
  tree_->Branch("event", &event_);
  tree_->Branch("lc_E", &lcE_);
  tree_->Branch("lc_x", &lcX_);
  tree_->Branch("lc_y", &lcY_);
  tree_->Branch("lc_z", &lcZ_);
  tree_->Branch("lc_layer", &lcLayer_);
  tree_->Branch("lc_side", &lcSide_);
  tree_->Branch("lc_det", &lcDet_);
  tree_->Branch("lc_nhits", &lcNHits_);
  tree_->Branch("lc_algo", &lcAlgo_);
  tree_->Branch("lc_seed", &lcSeed_);
  tree_->Branch("lc_mask", &lcMask_);
  tree_->Branch("lc_noTruthE", &lcNoTruthE_);
  tree_->Branch("lc_recE", &lcRecE_);
  tree_->Branch("lc_missingRecHits", &lcMissingRecHits_);
  tree_->Branch("tr_lc", &trLC_);
  tree_->Branch("tr_bp", &trBP_);
  tree_->Branch("tr_E", &trE_);
  tree_->Branch("rh_E", &rhE_);
  tree_->Branch("rh_x", &rhX_);
  tree_->Branch("rh_y", &rhY_);
  tree_->Branch("rh_z", &rhZ_);
  tree_->Branch("rh_layer", &rhLayer_);
  tree_->Branch("rh_det", &rhDet_);
  tree_->Branch("rh_lc", &rhLC_);
  tree_->Branch("trh_rh", &trhRH_);
  tree_->Branch("trh_bp", &trhBP_);
  tree_->Branch("trh_E", &trhE_);
  tree_->Branch("bp_id", &bpId_);
  tree_->Branch("bp_pdg", &bpPdg_);
  tree_->Branch("bp_signal", &bpSignal_);
  tree_->Branch("bp_bx", &bpBX_);
  tree_->Branch("bp_evt", &bpEvt_);
  tree_->Branch("bp_kind", &bpKind_);
  tree_->Branch("bp_origin", &bpOrigin_);
  tree_->Branch("bp_originPdg", &bpOriginPdg_);
  tree_->Branch("bp_originE", &bpOriginE_);
  tree_->Branch("bp_E", &bpE_);
  tree_->Branch("bp_eta", &bpEta_);
  tree_->Branch("bp_phi", &bpPhi_);
  tree_->Branch("bp_simE", &bpSimE_);
  if (!assignmentTag_.label().empty())
    tree_->Branch("clue_assignment", &clueAssignment_);
  for (std::size_t s = 0; s < trackstersTags_.size(); ++s) {
    const std::string n = "ts_" + trackstersTags_[s].label();
    tree_->Branch((n + "_E").c_str(), &tracksters_[s].energy);
    tree_->Branch((n + "_nLC").c_str(), &tracksters_[s].nLC);
    tree_->Branch((n + "_lc").c_str(), &tracksters_[s].lc);
    tree_->Branch((n + "_mult").c_str(), &tracksters_[s].mult);
  }
}

void TracksterTruthNtuplizer::analyze(edm::Event const& event, edm::EventSetup const& setup) {
  rhtools_.setGeometry(setup.getData(geomToken_));
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& lcs = event.get(layerClustersToken_);
  auto const& mask = event.get(maskToken_);
  const uint32_t nP = graph.nParticles();
  run_ = event.id().run();
  lumi_ = event.id().luminosityBlock();
  event_ = event.id().event();

  // --- base particle and origin of every truth particle -------------------------------------------
  std::vector<int> parent(nP);
  for (uint32_t p = 0; p < nP; ++p)
    parent[p] = firstParent(graph, p);
  auto crossed = [&](uint32_t p) {
    return !graph.particles()[p].backscattered && truth::Particle(&graph, p).checkpoint(0).has_value();
  };
  auto nearestCrossing = [&](uint32_t p) {
    for (int u = static_cast<int>(p); u >= 0; u = parent[u])
      if (crossed(u))
        return u;
    return kNone;
  };
  auto nearestAt = [&](uint32_t p, truth::LevelFlag flag) {
    for (int u = static_cast<int>(p); u >= 0; u = parent[u])
      if (graph.particles()[u].isAtLevel(flag))
        return u;
    return kNone;
  };
  auto rootOf = [&](uint32_t p) {
    int u = static_cast<int>(p);
    while (parent[u] >= 0)
      u = parent[u];
    return u;
  };
  std::vector<int> baseOf(nP, kNone), kindOf(nP, kRoot);
  for (uint32_t p = 0; p < nP; ++p) {
    if (int b = crossing_ ? nearestCrossing(p) : nearestAt(p, truth::LevelFlag::CaloBoundary); b >= 0) {
      baseOf[p] = b;
      kindOf[p] = kCaloBoundary;
    } else if (int r = nearestAt(p, truth::LevelFlag::ReconstructableFinalState); r >= 0) {
      baseOf[p] = r;
      kindOf[p] = kReconstructableFinalState;
    } else {
      baseOf[p] = rootOf(p);
      kindOf[p] = kRoot;
    }
  }

  // --- in-time sim energy per cell, per base particle ---------------------------------------------
  std::unordered_map<int, int> bpIndex;  // graph particle id -> row of the base-particle table
  std::vector<double> bpSim;
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellSim;
  std::unordered_map<uint32_t, double> cellTotal;
  for (uint32_t p = 0; p < nP; ++p) {
    auto hits = hitIndex.directHits(truth::HitChannel::Calo, p);
    if (hits.empty())
      continue;
    const int b = baseOf[p];
    auto [it, inserted] = bpIndex.emplace(b, static_cast<int>(bpIndex.size()));
    if (inserted)
      bpSim.push_back(0.);
    const int row = it->second;
    for (auto const& h : hits) {
      if (h.energy <= 0.f)
        continue;
      cellSim[h.detId].emplace_back(row, h.energy);
      cellTotal[h.detId] += h.energy;
      bpSim[row] += h.energy;
    }
  }

  std::unordered_map<uint32_t, float> recHitEnergy;
  for (auto const& token : recHitTokens_)
    for (auto const& rh : event.get(token))
      recHitEnergy[rh.detid().rawId()] = rh.energy();

  // --- layer clusters and the truth table --------------------------------------------------------
  for (auto* v : {&lcE_, &lcX_, &lcY_, &lcZ_, &lcMask_, &lcNoTruthE_, &lcRecE_, &trE_})
    v->clear();
  for (auto* v : {&lcLayer_, &lcSide_, &lcDet_, &lcNHits_, &lcAlgo_, &lcMissingRecHits_, &trLC_, &trBP_})
    v->clear();
  lcSeed_.clear();
  std::unordered_map<uint32_t, std::pair<int, float>> rhBestLC;  // rechit -> layer cluster with the largest fraction
  std::vector<double> accum(bpIndex.size(), 0.);
  std::vector<int> touched;
  for (std::size_t l = 0; l < lcs.size(); ++l) {
    auto const& lc = lcs[l];
    const DetId seed = lc.seed();
    lcE_.push_back(lc.energy());
    lcX_.push_back(lc.x());
    lcY_.push_back(lc.y());
    lcZ_.push_back(lc.z());
    lcLayer_.push_back(static_cast<int>(rhtools_.getLayerWithOffset(seed)));
    lcSide_.push_back(lc.z() > 0 ? 1 : 0);
    lcDet_.push_back(static_cast<int>(seed.det()));
    lcNHits_.push_back(static_cast<int>(lc.hitsAndFractions().size()));
    lcAlgo_.push_back(static_cast<int>(lc.algo()));
    lcSeed_.push_back(seed.rawId());
    lcMask_.push_back(l < mask.size() ? mask[l] : 0.f);

    double noTruth = 0., recSum = 0.;
    int missing = 0;
    touched.clear();
    for (auto const& [detId, fraction] : lc.hitsAndFractions()) {
      auto rh = recHitEnergy.find(detId.rawId());
      if (rh == recHitEnergy.end()) {
        ++missing;
        continue;
      }
      const double share = static_cast<double>(fraction) * rh->second;
      recSum += share;
      if (auto [b, ins] = rhBestLC.emplace(detId.rawId(), std::make_pair(static_cast<int>(l), fraction));
          !ins && fraction > b->second.second)
        b->second = {static_cast<int>(l), fraction};
      auto tot = cellTotal.find(detId.rawId());
      if (tot == cellTotal.end() || tot->second <= 0.) {
        noTruth += share;
        continue;
      }
      for (auto const& [row, e] : cellSim[detId.rawId()]) {
        if (accum[row] == 0.)
          touched.push_back(row);
        accum[row] += share * e / tot->second;
      }
    }
    std::sort(touched.begin(), touched.end());
    for (int row : touched) {
      trLC_.push_back(static_cast<int>(l));
      trBP_.push_back(row);
      trE_.push_back(static_cast<float>(accum[row]));
      accum[row] = 0.;
    }
    lcNoTruthE_.push_back(static_cast<float>(noTruth));
    lcRecE_.push_back(static_cast<float>(recSum));
    lcMissingRecHits_.push_back(missing);
  }

  // --- rechits in layer clusters and their truth table (sorted by DetId: reproducible order) ------
  for (auto* v : {&rhE_, &rhX_, &rhY_, &rhZ_, &trhE_})
    v->clear();
  for (auto* v : {&rhLayer_, &rhDet_, &rhLC_, &trhRH_, &trhBP_})
    v->clear();
  std::vector<uint32_t> rhIds;
  rhIds.reserve(rhBestLC.size());
  for (auto const& [id, best] : rhBestLC)
    rhIds.push_back(id);
  std::sort(rhIds.begin(), rhIds.end());
  for (uint32_t id : rhIds) {
    const DetId det(id);
    const float e = recHitEnergy[id];
    const auto pos = rhtools_.getPosition(det);
    const int row = static_cast<int>(rhE_.size());
    rhE_.push_back(e);
    rhX_.push_back(pos.x());
    rhY_.push_back(pos.y());
    rhZ_.push_back(pos.z());
    rhLayer_.push_back(static_cast<int>(rhtools_.getLayerWithOffset(det)));
    rhDet_.push_back(static_cast<int>(det.det()));
    rhLC_.push_back(rhBestLC[id].first);
    auto tot = cellTotal.find(id);
    if (tot == cellTotal.end() || tot->second <= 0.)
      continue;
    touched.clear();
    for (auto const& [b, s] : cellSim[id]) {
      if (accum[b] == 0.)
        touched.push_back(b);
      accum[b] += static_cast<double>(e) * s / tot->second;
    }
    std::sort(touched.begin(), touched.end());
    for (int b : touched) {
      trhRH_.push_back(row);
      trhBP_.push_back(b);
      trhE_.push_back(static_cast<float>(accum[b]));
      accum[b] = 0.;
    }
  }

  // --- base-particle table --------------------------------------------------------------------------
  for (auto* v : {&bpId_, &bpPdg_, &bpSignal_, &bpBX_, &bpEvt_, &bpKind_, &bpOrigin_, &bpOriginPdg_})
    v->assign(bpIndex.size(), kNone);
  for (auto* v : {&bpE_, &bpEta_, &bpPhi_, &bpSimE_, &bpOriginE_})
    v->assign(bpIndex.size(), 0.f);
  for (auto const& [gid, row] : bpIndex) {
    auto const& pd = graph.particles()[gid];
    bpId_[row] = gid;
    bpPdg_[row] = pd.pdgId;
    bpSignal_[row] = pd.isSignal() ? 1 : 0;
    bpBX_[row] = pd.bunchCrossing();
    bpEvt_[row] = pd.eventIndex();
    bpKind_[row] = (crossing_ ? crossed(gid) : pd.isAtLevel(truth::LevelFlag::CaloBoundary)) ? kCaloBoundary
                   : pd.isAtLevel(truth::LevelFlag::ReconstructableFinalState)               ? kReconstructableFinalState
                                                                                             : kRoot;
    // energy and direction where it entered the calorimeter, if recorded
    if (auto cp = truth::Particle(&graph, gid).checkpoint(0)) {
      bpE_[row] = cp->momentum.energy();
      bpEta_[row] = cp->position.eta();
      bpPhi_[row] = cp->position.phi();
    } else {
      bpE_[row] = pd.momentum.energy();
      bpEta_[row] = pd.momentum.pt() > 0 ? pd.momentum.eta() : 0.f;
      bpPhi_[row] = pd.momentum.phi();
    }
    bpSimE_[row] = static_cast<float>(bpSim[row]);
    if (int o = nearestAt(gid, truth::LevelFlag::ReconstructableFinalState); o >= 0) {
      bpOrigin_[row] = o;
      bpOriginPdg_[row] = graph.particles()[o].pdgId;
      bpOriginE_[row] = graph.particles()[o].momentum.energy();
    }
  }

  // --- CMSSW reference ----------------------------------------------------------------------------
  if (!assignmentTag_.label().empty()) {
    auto const& a = event.get(assignmentToken_);
    clueAssignment_.assign(a.begin(), a.end());
  }
  for (std::size_t s = 0; s < trackstersTokens_.size(); ++s) {
    auto& out = tracksters_[s];
    out.energy.clear();
    out.nLC.clear();
    out.lc.clear();
    out.mult.clear();
    for (auto const& t : event.get(trackstersTokens_[s])) {
      out.energy.push_back(t.raw_energy());
      out.nLC.push_back(static_cast<int>(t.vertices().size()));
      for (std::size_t k = 0; k < t.vertices().size(); ++k) {
        out.lc.push_back(static_cast<int>(t.vertices()[k]));
        out.mult.push_back(k < t.vertex_multiplicity().size() ? t.vertex_multiplicity()[k] : 1.f);
      }
    }
  }
  tree_->Fill();
}

void TracksterTruthNtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<std::string>("baseLevel", "crossing")
      ->setComment("crossing: hits go to the nearest ancestor that crossed into the calorimeter (every crossing); "
                   "caloBoundary: to the nearest member of the caloBoundary level (outermost crossings only)");
  desc.add<edm::InputTag>("layerClusterMask", edm::InputTag("filteredLayerClustersCLUE3DHigh", "CLUE3DHigh"))
      ->setComment("The mask of the trackster-building iteration: the layer clusters it may use");
  desc.add<edm::InputTag>("clueAssignment", edm::InputTag("ticlTrackstersCLUEsteringAssignment"))
      ->setComment("CLUEstering trackster index per layer cluster (-1 = outlier or masked); empty label = skip");
  desc.add<std::vector<edm::InputTag>>("tracksters", {edm::InputTag("ticlTrackstersCLUE3DHigh")});
  desc.add<std::vector<edm::InputTag>>("recHits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TracksterTruthNtuplizer);
