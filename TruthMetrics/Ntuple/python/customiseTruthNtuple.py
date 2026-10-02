# Attach the trackster-building truth ntuple to a RECO (step3) process.
#
#   customiseTruthNtuple            the truth graph and hit index come from the input
#                                   (DIGI run with --procModifiers enableTruth): pileup-aware
#   customiseTruthNtupleAndGraphJson  the same, plus the truth graph as JSON per event (0 PU display)
#   customiseTruthNtupleBuildTruth  build them here from the signal g4SimHits: no-pileup samples
#                                   whose DIGI did not run enableTruth
#
# Run step3 with --procModifiers ticl_dev so that the CLUE3DHigh step uses CLUEstering and the
# CLUEstering assignment product exists. Output: truthNtuple.root (TFileService).
import FWCore.ParameterSet.Config as cms


def _addNtuple(process, fileName):
    from TruthMetrics.Ntuple.tracksterTruthNtuplizer_cfi import tracksterTruthNtuplizer
    process.tracksterTruthNtuplizer = tracksterTruthNtuplizer.clone()
    if not hasattr(process, "TFileService"):
        process.TFileService = cms.Service("TFileService", fileName=cms.string(fileName))
    process.truthNtuple_step = cms.EndPath(process.tracksterTruthNtuplizer)
    if process.schedule is not None:
        process.schedule.append(process.truthNtuple_step)
    return process


def customiseTruthNtuple(process, fileName="truthNtuple.root"):
    return _addNtuple(process, fileName)


def customiseTruthNtupleBuildTruth(process, fileName="truthNtuple.root", fragment="SinglePiPt25Eta1p7_2p7"):
    from PhysicsTools.TruthInfo.modules import (TruthGraphProducer, TruthLogicalGraphProducer,
                                                TruthLogicalGraphHitIndexProducer)
    from PhysicsTools.TruthInfo.truthGraphSelections import postProcessingPSet
    process.truthGraphProducer = TruthGraphProducer()
    process.truthLogicalGraphProducer = TruthLogicalGraphProducer(postProcessing=postProcessingPSet(fragment))
    process.truthLogicalGraphHitIndexProducer = TruthLogicalGraphHitIndexProducer(
        src="truthLogicalGraphProducer",
        rawSrc="truthGraphProducer",
        recHitMap=cms.InputTag(""),
        simHitCollections=[
            cms.InputTag("g4SimHits", "HGCHitsEE"),
            cms.InputTag("g4SimHits", "HGCHitsHEfront"),
            cms.InputTag("g4SimHits", "HGCHitsHEback"),
            cms.InputTag("g4SimHits", "EcalHitsEB"),
            cms.InputTag("g4SimHits", "HcalHits"),
        ],
        doHGCalRelabelling=False,
    )
    process.truthNtupleTruthTask = cms.Task(process.truthGraphProducer, process.truthLogicalGraphProducer,
                                            process.truthLogicalGraphHitIndexProducer)
    _addNtuple(process, fileName)
    process.truthNtuple_step.associate(process.truthNtupleTruthTask)
    return process


def customiseTruthNtupleAndGraphJson(process, fileName="truthNtuple.root", jsonPrefix="truthgraph"):
    """customiseTruthNtuple, plus the truth graph of every event as JSON (TruthMetrics/Display
    TruthGraphJsonDumper): particles, vertices, boundary crossings and direct sim hits, no reco.
    Particle ids are the graph indices, the same as the ntuple's bp_id."""
    _addNtuple(process, fileName)
    process.truthGraphJsonDumper = cms.EDAnalyzer(
        "TruthGraphJsonDumper",
        src=cms.InputTag("truthLogicalGraphProducer"),
        hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
        layerClusters=cms.InputTag(""),
        tracksters=cms.VInputTag(),
        outPrefix=cms.string(jsonPrefix),
        minHitEnergy=cms.double(0.),
        roiDeltaR=cms.double(0.),
    )
    process.truthNtuple_step += process.truthGraphJsonDumper
    return process
