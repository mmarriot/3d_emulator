// Writes one JSON file per event for the browser event display in TruthMetrics/Display/viewer:
// the logical truth graph (particles, vertices, level flags, pileup provenance), each particle's
// direct calorimeter sim hits, and optionally the layer clusters and trackster collections, all
// referring to one shared table of cells with their positions.
//
// Particle "id" is the index in this file; "gid" is the index in the event's truth graph (= the ntuple's bp_id).
// With roiDeltaR > 0 only cells within that cone around the signal particle (same endcap) are
// written, and only particles from the signal interaction or with hits in the cone (plus their
// ancestors, so the graph stays connected). That is what keeps a 200-PU event viewable.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/VertexData.h"

class TruthGraphJsonDumper : public edm::one::EDAnalyzer<> {
public:
  explicit TruthGraphJsonDumper(edm::ParameterSet const&);
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const edm::InputTag layerClustersTag_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  const std::vector<edm::InputTag> trackstersTags_;
  const std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> trackstersTokens_;
  const std::string outPrefix_;
  const double minHitEnergy_;
  const double roiDeltaR_;
};

TruthGraphJsonDumper::TruthGraphJsonDumper(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      layerClustersTag_(cfg.getParameter<edm::InputTag>("layerClusters")),
      trackstersTags_(cfg.getParameter<std::vector<edm::InputTag>>("tracksters")),
      trackstersTokens_(edm::vector_transform(
          trackstersTags_, [this](edm::InputTag const& t) { return consumes<std::vector<ticl::Trackster>>(t); })),
      outPrefix_(cfg.getParameter<std::string>("outPrefix")),
      minHitEnergy_(cfg.getParameter<double>("minHitEnergy")),
      roiDeltaR_(cfg.getParameter<double>("roiDeltaR")) {
  if (!layerClustersTag_.label().empty())
    layerClustersToken_ = consumes<std::vector<reco::CaloCluster>>(layerClustersTag_);
}

void TruthGraphJsonDumper::analyze(edm::Event const& event, edm::EventSetup const& setup) {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& geom = setup.getData(geomToken_);
  const uint32_t nP = graph.nParticles();

  auto isSignal = [&](uint32_t p) { return graph.particles()[p].isSignal(); };

  // ROI axis: the most energetic signal GEN particle.
  double roiEta = 0, roiPhi = 0, best = -1;
  for (uint32_t p = 0; p < nP; ++p) {
    auto const& pd = graph.particles()[p];
    if (pd.hasGen() && isSignal(p) && pd.momentum.pt() > 0 && pd.momentum.energy() > best) {
      best = pd.momentum.energy();
      roiEta = pd.momentum.eta();
      roiPhi = pd.momentum.phi();
    }
  }
  auto inRoi = [&](GlobalPoint const& x) {
    if (roiDeltaR_ <= 0)
      return true;
    return x.z() * roiEta > 0 && reco::deltaR(x.eta(), x.phi(), roiEta, roiPhi) < roiDeltaR_;
  };

  // Shared cell table: index, position, detector.
  std::unordered_map<uint32_t, int32_t> cellIndex;  // -1 = outside the ROI or no geometry
  std::vector<GlobalPoint> cellPos;
  std::vector<uint32_t> cellIds;
  auto cellOf = [&](uint32_t rawId) -> int32_t {
    auto it = cellIndex.find(rawId);
    if (it != cellIndex.end())
      return it->second;
    int32_t idx = -1;
    DetId id(rawId);
    auto const* subGeom = geom.getSubdetectorGeometry(id);
    if (subGeom != nullptr && subGeom->getGeometry(id) != nullptr) {
      auto pos = geom.getPosition(id);
      if (inRoi(pos)) {
        idx = cellPos.size();
        cellPos.push_back(pos);
        cellIds.push_back(rawId);
      }
    }
    cellIndex.emplace(rawId, idx);
    return idx;
  };

  // Hits per particle, then which particles to keep: signal, or with ROI hits, plus ancestors.
  std::vector<std::vector<std::pair<int32_t, float>>> hits(nP);
  std::vector<uint8_t> keep(nP, 0);
  for (uint32_t p = 0; p < nP; ++p) {
    for (auto const& h : hitIndex.directHits(truth::HitChannel::Calo, p)) {
      if (h.energy < minHitEnergy_)
        continue;
      if (auto c = cellOf(h.detId); c >= 0)
        hits[p].emplace_back(c, h.energy);
    }
    if (roiDeltaR_ <= 0 || isSignal(p) || !hits[p].empty())
      keep[p] = 1;
  }
  std::vector<uint32_t> stack;
  for (uint32_t p = 0; p < nP; ++p)
    if (keep[p])
      stack.push_back(p);
  while (!stack.empty()) {
    uint32_t p = stack.back();
    stack.pop_back();
    for (auto v : graph.productionVertices(p))
      for (auto q : graph.incomingParticles(v))
        if (!keep[q]) {
          keep[q] = 1;
          stack.push_back(q);
        }
  }
  std::vector<int32_t> newP(nP, -1), newV(graph.nVertices(), -1);
  std::vector<uint32_t> keptP, keptV;
  for (uint32_t p = 0; p < nP; ++p)
    if (keep[p]) {
      newP[p] = keptP.size();
      keptP.push_back(p);
      for (auto v : graph.productionVertices(p))
        if (newV[v] < 0) {
          newV[v] = keptV.size();
          keptV.push_back(v);
        }
      for (auto v : graph.decayVertices(p))
        if (newV[v] < 0) {
          newV[v] = keptV.size();
          keptV.push_back(v);
        }
    }

  std::ofstream out(outPrefix_ + "_" + std::to_string(event.id().event()) + ".json");
  out << std::setprecision(5);
  out << "{\"run\":" << event.id().run() << ",\"event\":" << event.id().event() << ",\"roiDeltaR\":" << roiDeltaR_
      << ",\"roiEta\":" << roiEta << ",\"roiPhi\":" << roiPhi << ",\"nParticlesTotal\":" << nP << ",\n";

  // Level names in bit order, so the viewer decodes ParticleData::levelFlags.
  out << "\"levels\":[";
  for (std::size_t i = 0; i < truth::kLevelTable.size(); ++i)
    out << (i ? "," : "") << "{\"name\":\"" << truth::kLevelTable[i].name
        << "\",\"bit\":" << static_cast<uint32_t>(truth::kLevelTable[i].flag) << "}";
  out << "],\n";

  out << "\"vertices\":[";
  for (std::size_t i = 0; i < keptV.size(); ++i) {
    auto const& vd = graph.vertices()[keptV[i]];
    out << (i ? "," : "") << "\n{\"id\":" << i << ",\"x\":" << vd.position.x() << ",\"y\":" << vd.position.y()
        << ",\"z\":" << vd.position.z() << ",\"role\":" << int(vd.role) << ",\"reason\":\""
        << truth::vertexReasonName(static_cast<truth::VertexReason>(vd.reason)) << "\"}";
  }
  out << "],\n";

  out << "\"particles\":[";
  for (std::size_t i = 0; i < keptP.size(); ++i) {
    const uint32_t p = keptP[i];
    auto const& pd = graph.particles()[p];
    auto const& mom = pd.momentum;
    // Pileup provenance: every particle, GEN-only included, carries its interaction.
    const int bx = pd.bunchCrossing(), evt = pd.eventIndex();
    out << (i ? "," : "") << "\n{\"id\":" << i << ",\"gid\":" << p << ",\"pdg\":" << pd.pdgId << ",\"E\":" << mom.energy()
        << ",\"pt\":" << mom.pt() << ",\"eta\":" << (mom.pt() > 0 ? mom.eta() : 0.) << ",\"phi\":" << mom.phi()
        << ",\"gen\":" << pd.hasGen() << ",\"sim\":" << pd.hasSim() << ",\"status\":" << pd.status
        << ",\"levels\":" << pd.levelFlags << ",\"bs\":" << pd.backscattered << ",\"role\":" << int(pd.role)
        << ",\"bx\":" << bx << ",\"evt\":" << evt << ",\"genEvent\":" << pd.genEvent;
    out << ",\"prod\":[";
    bool first = true;
    for (auto v : graph.productionVertices(p)) {
      out << (first ? "" : ",") << newV[v];
      first = false;
    }
    out << "],\"dec\":[";
    first = true;
    for (auto v : graph.decayVertices(p)) {
      out << (first ? "" : ",") << newV[v];
      first = false;
    }
    out << "]";
    if (auto cp = truth::Particle(&graph, p).checkpoint(0))
      out << ",\"cb\":[" << cp->position.x() << "," << cp->position.y() << "," << cp->position.z() << ","
          << cp->momentum.energy() << "]";
    out << ",\"hits\":[";
    for (std::size_t k = 0; k < hits[p].size(); ++k)
      out << (k ? "," : "") << "[" << hits[p][k].first << "," << hits[p][k].second << "]";
    out << "]}";
  }
  out << "],\n";

  // Layer clusters inside the ROI (by position), each with its [cell, fraction] list.
  std::vector<int32_t> newLC;
  out << "\"layerClusters\":[";
  if (!layerClustersTag_.label().empty()) {
    auto const& lcs = event.get(layerClustersToken_);
    newLC.assign(lcs.size(), -1);
    int32_t n = 0;
    for (std::size_t i = 0; i < lcs.size(); ++i) {
      auto const& lc = lcs[i];
      GlobalPoint pos(lc.x(), lc.y(), lc.z());
      if (!inRoi(pos))
        continue;
      newLC[i] = n;
      out << (n ? "," : "") << "\n{\"x\":" << lc.x() << ",\"y\":" << lc.y() << ",\"z\":" << lc.z()
          << ",\"E\":" << lc.energy() << ",\"hits\":[";
      bool first = true;
      for (auto const& [detId, fraction] : lc.hitsAndFractions()) {
        auto c = cellOf(detId.rawId());
        if (c < 0)
          continue;
        out << (first ? "" : ",") << "[" << c << "," << fraction << "]";
        first = false;
      }
      out << "]}";
      ++n;
    }
  }
  out << "],\n";

  // Tracksters: kept if any of their layer clusters is in the ROI; eRoi is the raw energy of
  // the layer clusters inside it (each counted with its trackster vertex multiplicity).
  out << "\"trackstersets\":[";
  for (std::size_t s = 0; s < trackstersTags_.size(); ++s) {
    auto const& tss = event.get(trackstersTokens_[s]);
    auto const& lcs = event.get(layerClustersToken_);
    out << (s ? "," : "") << "\n{\"name\":\"" << trackstersTags_[s].label() << "\",\"tracksters\":[";
    bool firstT = true;
    for (auto const& t : tss) {
      std::vector<std::pair<int32_t, float>> in;
      double eRoi = 0;
      for (std::size_t k = 0; k < t.vertices().size(); ++k) {
        auto lc = t.vertices()[k];
        if (lc < newLC.size() && newLC[lc] >= 0) {
          const float mult = k < t.vertex_multiplicity().size() ? t.vertex_multiplicity()[k] : 1.f;
          in.emplace_back(newLC[lc], mult);
          eRoi += lcs[lc].energy() / (mult > 0 ? mult : 1.f);
        }
      }
      if (in.empty())
        continue;
      out << (firstT ? "" : ",") << "\n{\"E\":" << t.raw_energy() << ",\"eRoi\":" << eRoi << ",\"nLC\":"
          << t.vertices().size() << ",\"lcs\":[";
      for (std::size_t k = 0; k < in.size(); ++k)
        out << (k ? "," : "") << "[" << in[k].first << "," << in[k].second << "]";
      out << "]}";
      firstT = false;
    }
    out << "]}";
  }
  out << "],\n";

  out << "\"cells\":[";
  for (std::size_t c = 0; c < cellPos.size(); ++c)
    out << (c ? "," : "") << "[" << cellPos[c].x() << "," << cellPos[c].y() << "," << cellPos[c].z() << ","
        << DetId(cellIds[c]).det() << "]";
  out << "]}\n";
}

void TruthGraphJsonDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"))
      ->setComment("Empty label = do not write layer clusters (and then no tracksters either)");
  desc.add<std::vector<edm::InputTag>>("tracksters",
                                       {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTracksterLinks")});
  desc.add<std::string>("outPrefix", "truthgraph");
  desc.add<double>("minHitEnergy", 0.)->setComment("Drop direct sim hits below this energy [GeV]");
  desc.add<double>("roiDeltaR", 0.)->setComment("If > 0, keep only cells within this cone around the signal");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TruthGraphJsonDumper);
