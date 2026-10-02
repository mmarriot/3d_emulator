# Build the logical truth graph + hit index from a GEN-SIM-RECO file and write one
# JSON per event for the browser event display (TruthMetrics/Display/viewer).
#   cmsRun dumpTruthGraphJson_cfg.py step3.root -o outdir -t default -f SinglePiPt25Eta1p7_2p7
# For a pileup sample produced with --procModifiers enableTruth (graph + hit index built during
# mixing and stored in the file), read those instead of rebuilding from the signal g4SimHits:
#   cmsRun dumpTruthGraphJson_cfg.py step3.root -o outdir -t PU200 --fromFile --roi 0.4
import os
from argparse import ArgumentParser

import FWCore.ParameterSet.Config as cms
from PhysicsTools.TruthInfo.modules import TruthGraphProducer, TruthLogicalGraphProducer, TruthLogicalGraphHitIndexProducer
from PhysicsTools.TruthInfo.truthGraphSelections import postProcessingPSet

parser = ArgumentParser()
parser.add_argument("inputFile")
parser.add_argument("-o", "--outdir", default=".")
parser.add_argument("-t", "--tag", default="")
parser.add_argument("-n", "--maxevts", type=int, default=-1)
parser.add_argument("-f", "--fragment", default="SinglePiPt25Eta1p7_2p7",
                    help="generator fragment; picks the truthGraphSelections preset")
parser.add_argument("--fromFile", action="store_true",
                    help="use the truth graph and hit index stored in the input (enableTruth at DIGI)")
parser.add_argument("--roi", type=float, default=0.,
                    help="keep only cells within this dR of the signal particle (0 = everything)")
parser.add_argument("--minHitEnergy", type=float, default=0., help="sim hit threshold [GeV]")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

process = cms.Process("TRUTHJSON")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.load("Configuration.Geometry.GeometryExtendedRun4D120Reco_cff")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxevts))
inp = args.inputFile if ":" in args.inputFile else "file:" + args.inputFile
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(inp))

process.truthGraphProducer = TruthGraphProducer()
process.truthLogicalGraphProducer = TruthLogicalGraphProducer(postProcessing=postProcessingPSet(args.fragment))

process.detIdToRecHitMapProducer = cms.EDProducer(
    "DetIdToRecHitMapProducer",
    hgcalRecHits=cms.VInputTag(
        cms.InputTag("HGCalRecHit", "HGCEERecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEFRecHits", "RECO"),
        cms.InputTag("HGCalRecHit", "HGCHEBRecHits", "RECO"),
    ),
    pfRecHits=cms.VInputTag(),
)
process.truthLogicalGraphHitIndexProducer = TruthLogicalGraphHitIndexProducer(
    src="truthLogicalGraphProducer",
    rawSrc="truthGraphProducer",
    recHitMap="detIdToRecHitMapProducer",
    simHitCollections=[
        cms.InputTag("g4SimHits", "HGCHitsEE", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront", "SIM"),
        cms.InputTag("g4SimHits", "HGCHitsHEback", "SIM"),
        cms.InputTag("g4SimHits", "EcalHitsEB", "SIM"),
        cms.InputTag("g4SimHits", "HcalHits", "SIM"),
    ],
    doHGCalRelabelling=False,
)

process.truthGraphJsonDumper = cms.EDAnalyzer(
    "TruthGraphJsonDumper",
    src=cms.InputTag("truthLogicalGraphProducer"),
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    layerClusters=cms.InputTag("hgcalMergeLayerClusters"),
    tracksters=cms.VInputTag("ticlTrackstersCLUE3DHigh", "ticlTracksterLinks"),
    outPrefix=cms.string(os.path.join(args.outdir, f"truthgraph{('_' + args.tag) if args.tag else ''}")),
    minHitEnergy=cms.double(args.minHitEnergy),
    roiDeltaR=cms.double(args.roi),
)

if args.fromFile:
    # Products from the input file. The producers must not exist in this process at all, or
    # they would run on demand under the same labels and shadow the stored products.
    for name in ("truthGraphProducer", "truthLogicalGraphProducer", "detIdToRecHitMapProducer",
                 "truthLogicalGraphHitIndexProducer"):
        delattr(process, name)
    process.p = cms.Path(process.truthGraphJsonDumper)
else:
    process.p = cms.Path(
        process.truthGraphProducer
        + process.truthLogicalGraphProducer
        + process.detIdToRecHitMapProducer
        + process.truthLogicalGraphHitIndexProducer
        + process.truthGraphJsonDumper
    )
