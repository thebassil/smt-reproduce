"""Model cards for cross-system ablation."""
from pipeline.models.base import ModelCard
from pipeline.models.greedy_logic import GreedyLogicCard
from pipeline.models.pwc import PWCCard
from pipeline.models.solver_logic_pwc import SolverLogicPWCCard
from pipeline.models.knn import KNNCard
from pipeline.models.instance_clustering import InstanceClusteringCard
from pipeline.models.lambdarank import LambdaRankCard
from pipeline.models.ordinal_regression import OrdinalRegressionCard
from pipeline.models.quantile_regression import QuantileRegressionCard
from pipeline.models.gp_regression import GPRegressionCard
from pipeline.models.cost_sensitive import CostSensitiveCard
from pipeline.models.autofolio import AutoFolioCard
from pipeline.models.survival import SurvivalCard
from pipeline.models.collaborative_filtering import CollaborativeFilteringCard
from pipeline.models.conformal import ConformalCard
from pipeline.models.mosap import MOSAPCard
from pipeline.models.online_learning import OnlineLearningCard
from pipeline.models.contextual_bandits import ContextualBanditsCard
from pipeline.models.stacking import StackingCard

# Optional imports (require torch)
try:
    from pipeline.models.deep_kernel import DeepKernelCard
    from pipeline.models.meta_learning import MetaLearningCard
    from pipeline.models.hypernetworks import HypernetworkCard
    from pipeline.models.moe import MoECard
    from pipeline.models.transformer import TransformerCard
    from pipeline.models.neural_processes import NeuralProcessCard
    from pipeline.models.rl_scheduling import RLSchedulingCard
except ImportError:
    pass

# Optional imports (require torch_geometric)
try:
    from pipeline.models.gin import GINCard
    from pipeline.models.gcn import GCNCard
    from pipeline.models.graphsage import GraphSAGECard
    from pipeline.models.graph_transformer import GraphTransformerCard
    from pipeline.models.gat_mlp import GATMLPCard
except ImportError:
    pass

__all__ = [
    "ModelCard",
    # Core
    "GreedyLogicCard",
    "PWCCard",
    "SolverLogicPWCCard",
    "KNNCard",
    "InstanceClusteringCard",
    "LambdaRankCard",
    # Predictive
    "OrdinalRegressionCard",
    "QuantileRegressionCard",
    "GPRegressionCard",
    "DeepKernelCard",
    "CostSensitiveCard",
    "AutoFolioCard",
    "SurvivalCard",
    # Ensemble / Meta-Learning
    "MetaLearningCard",
    "HypernetworkCard",
    "MoECard",
    "StackingCard",
    "TransformerCard",
    "NeuralProcessCard",
    # GNN
    "GINCard",
    "GCNCard",
    "GraphSAGECard",
    "GraphTransformerCard",
    "GATMLPCard",
    # Collaborative / Uncertainty
    "CollaborativeFilteringCard",
    "ConformalCard",
    "MOSAPCard",
    # Sequential / Online
    "OnlineLearningCard",
    "ContextualBanditsCard",
    "RLSchedulingCard",
]
