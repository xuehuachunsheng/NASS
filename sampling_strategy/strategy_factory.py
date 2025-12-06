from sampling_strategy.random_sampling import RandomSelectionStrategy
from sampling_strategy.gass_random_sampling import GASSRandomSelectionStrategy
from sampling_strategy.entropy_sampling import EntropyStrategy
from sampling_strategy.gass_entropy_sampling import GASSEntropyStrategy
from sampling_strategy.leastconfidence_sampling import LeastConfidenceStrategy
from sampling_strategy.gass_leastconfidence_sampling import GASSLeastConfidenceStrategy
from sampling_strategy.margin_sampling import MarginStrategy
from sampling_strategy.gass_margin_sampling import GASSMarginStrategy
from sampling_strategy.variance_reduction_sampling import VarianceReductionStrategy
from sampling_strategy.gass_variance_reduction_sampling import GASSVarianceReductionStrategy
from sampling_strategy.expected_model_change_sampling import ExpectedModelChangeStrategy
from sampling_strategy.gass_expected_model_change_sampling import GASSExpectedModelChangeStrategy
from sampling_strategy.lebgp_entropy_sampling import LEBGPEntropyStrategy
from sampling_strategy.lebgp_leastconfidence_sampling import LEBGPLeastConfidenceStrategy
from sampling_strategy.lebgp_margin_sampling import LEBGPMarginStrategy
from sampling_strategy.nass_random_sampling import NASSRandomSelectionStrategy
from sampling_strategy.nass_entropy_sampling import NASSEntropyStrategy
from sampling_strategy.nass_leastconfidence_sampling import NASSLeastConfidenceStrategy
from sampling_strategy.nass_margin_sampling import NASSMarginStrategy
from sampling_strategy.nass_lebgp_entropy_sampling import NASSLEBGPEntropyStrategy
from sampling_strategy.nass_lebgp_leastconfidence_sampling import NASSLEBGPLeastConfidenceStrategy
from sampling_strategy.nass_lebgp_margin_sampling import NASSLEBGPMarginStrategy

class BatchSampleSelectionStratgyFactory:

    def get_strategy(strategy_name, train_ds, labeled_idx, budget, **kwargs):
        if strategy_name == "random":
            return RandomSelectionStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "entropy":
            return EntropyStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "leastconfidence":
            return LeastConfidenceStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "margin":
            return MarginStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "vr":
            return VarianceReductionStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "emc":
            return ExpectedModelChangeStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "lebgp-entropy":
            return LEBGPEntropyStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "lebgp-leastconfidence":
            return LEBGPLeastConfidenceStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "lebgp-margin":
            return LEBGPMarginStrategy(train_ds, labeled_idx, budget, **{**kwargs})

    def get_gass_strategy(strategy_name, train_ds, labeled_idx, budget, **kwargs):
        if strategy_name == "random":
            return GASSRandomSelectionStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "entropy":
            return GASSEntropyStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "leastconfidence":
            return GASSLeastConfidenceStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "margin":
            return GASSMarginStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "vr":
            return GASSVarianceReductionStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "emc":
            return GASSExpectedModelChangeStrategy(train_ds, labeled_idx, budget, **{**kwargs})
    
    def get_nass_strategy(strategy_name, train_ds, labeled_idx, budget, **kwargs):
        if strategy_name == "random":
            return NASSRandomSelectionStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "entropy":
            return NASSEntropyStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "leastconfidence":
            return NASSLeastConfidenceStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "margin":
            return NASSMarginStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "lebgp-entropy":
            return NASSLEBGPEntropyStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "lebgp-leastconfidence":
            return NASSLEBGPLeastConfidenceStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        elif strategy_name == "lebgp-margin":
            return NASSLEBGPMarginStrategy(train_ds, labeled_idx, budget, **{**kwargs})
        
        