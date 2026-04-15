"""Featuriser card implementations for the cross-system ablation study.

All 32 featuriser cards, organized by category:
  VECTOR Tabular (2): StaticLightBoW, StaticExpanded
  VECTOR Learned (6): TransformerEmbed, ContrastiveEmbed, TreeLSTM, ASTPath, Formula2Vec, LLMEmbed
  VECTOR Kernel/Statistical (5): Entropy, WLKernel, ProofComplexity, HashKernel, RandomWalk
  VECTOR Graph-Derived (5): Community, StructuralWidth, Spectral, TDA, HypergraphFeatures
  VECTOR Dynamic (1): DynamicProbes
  GRAPH Heterogeneous (7): LCG*, VCG*, SMT-DAG, Tripartite, NeuroBack, AIG, TESS
  GRAPH Homogeneous (2): VIG, LIG
  GRAPH Hierarchical (1): SubgraphPool
  GRAPH Bipartite (2): VCG, LCG
  GRAPH Hypergraph (1): ClauseHypergraph
"""

# VECTOR: Tabular
from pipeline.featurisers.static_light_bow import StaticLightBoW
from pipeline.featurisers.static_expanded import StaticExpanded

# VECTOR: Learned Embeddings
try:
    from pipeline.featurisers.transformer_embed import TransformerEmbedFeaturiser
    from pipeline.featurisers.contrastive_embed import ContrastiveEmbedFeaturiser
    from pipeline.featurisers.tree_lstm import TreeLSTMFeaturiser
    from pipeline.featurisers.ast_path import ASTPathFeaturiser
    from pipeline.featurisers.formula2vec import Formula2VecFeaturiser
    from pipeline.featurisers.llm_embed import LLMEmbedFeaturiser
except ImportError:
    pass

# VECTOR: Kernel & Statistical
from pipeline.featurisers.entropy_features import EntropyFeatures
from pipeline.featurisers.wl_kernel import WLKernel
from pipeline.featurisers.proof_complexity import ProofComplexity
from pipeline.featurisers.hash_kernel import HashKernel
from pipeline.featurisers.random_walk_kernel import RandomWalkKernel

# VECTOR: Graph-Derived
from pipeline.featurisers.community_structure import CommunityStructure
from pipeline.featurisers.structural_width import StructuralWidth
from pipeline.featurisers.spectral_features import SpectralFeatures
from pipeline.featurisers.tda_features import TDAFeatures
from pipeline.featurisers.hypergraph_features import HypergraphFeatures

# VECTOR: Dynamic
from pipeline.featurisers.dynamic_probes import DynamicProbesFeaturiser

# GRAPH: Heterogeneous
try:
    from pipeline.featurisers.graph_lcg_star import LCGStarFeaturiser
    from pipeline.featurisers.graph_vcg_star import VCGStarFeaturiser
    from pipeline.featurisers.graph_smt_dag import SMTDAGFeaturiser
    from pipeline.featurisers.graph_tripartite import TripartiteGraphFeaturiser
    from pipeline.featurisers.graph_neuroback import NeuroBackFeaturiser
    from pipeline.featurisers.graph_aig import AIGFeaturiser
    from pipeline.featurisers.graph_tess import TESSFeaturiser

    # GRAPH: Homogeneous
    from pipeline.featurisers.graph_vig import VIGFeaturiser
    from pipeline.featurisers.graph_lig import LIGFeaturiser

    # GRAPH: Hierarchical
    from pipeline.featurisers.graph_subgraph_pool import SubgraphPoolFeaturiser

    # GRAPH: Bipartite
    from pipeline.featurisers.graph_vcg import VCGFeaturiser
    from pipeline.featurisers.graph_lcg import LCGFeaturiser

    # GRAPH: Hypergraph
    from pipeline.featurisers.graph_hypergraph import HypergraphFeaturiser
except ImportError:
    pass

# GRAPH: AST+UD (REF-GNN reference)
try:
    from pipeline.featurisers.sibyl_ast_ud import SibylASTUDFeaturiser
except ImportError:
    pass

__all__ = [
    # VECTOR: Tabular
    "StaticLightBoW", "StaticExpanded",
    # VECTOR: Learned Embeddings
    "TransformerEmbedFeaturiser", "ContrastiveEmbedFeaturiser",
    "TreeLSTMFeaturiser", "ASTPathFeaturiser",
    "Formula2VecFeaturiser", "LLMEmbedFeaturiser",
    # VECTOR: Kernel & Statistical
    "EntropyFeatures", "WLKernel", "ProofComplexity",
    "HashKernel", "RandomWalkKernel",
    # VECTOR: Graph-Derived
    "CommunityStructure", "StructuralWidth", "SpectralFeatures",
    "TDAFeatures", "HypergraphFeatures",
    # VECTOR: Dynamic
    "DynamicProbesFeaturiser",
    # GRAPH: Heterogeneous
    "LCGStarFeaturiser", "VCGStarFeaturiser", "SMTDAGFeaturiser",
    "TripartiteGraphFeaturiser", "NeuroBackFeaturiser",
    "AIGFeaturiser", "TESSFeaturiser",
    # GRAPH: Homogeneous
    "VIGFeaturiser", "LIGFeaturiser",
    # GRAPH: Hierarchical
    "SubgraphPoolFeaturiser",
    # GRAPH: Bipartite
    "VCGFeaturiser", "LCGFeaturiser",
    # GRAPH: Hypergraph
    "HypergraphFeaturiser",
    # GRAPH: AST+UD (REF-GNN)
    "SibylASTUDFeaturiser",
]
