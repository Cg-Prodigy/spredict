import numpy as np
import pandas as pd
from scipy.stats import skellam
from statsmodels.base.model import GenericLikelihoodModel

class SkellamModel(GenericLikelihoodModel):
    """
    Skellam Regression Model for predicting the difference between two Poisson-distributed variables
    (e.g., Home Goals - Away Goals).
    """
    def __init__(self, endog, exog, **kwargs):
        super(SkellamModel, self).__init__(endog, exog, **kwargs)

    def nloglikeobs(self, params):
        """
        Negative log-likelihood of the observations.
        """
        mu1 = np.exp(np.dot(self.exog, params))  # Expected home goals (lambda1)
        # For Skellam, we typically model mu1 and mu2 separately or model the difference directly.
        # However, standard Skellam regression usually assumes mu1 and mu2 are functions of covariates.
        # Here, we'll simplify:
        # We assume the goal difference Y = Home - Away follows Skellam(mu1, mu2).
        # But we need to map exog to mu1 and mu2.
        # A common approach is to have separate coefficients for home and away strengths,
        # but here we are predicting the *difference* directly based on interaction features.
        
        # Alternative approach:
        # We can treat the problem as predicting the difference directly, but Skellam parameters are means.
        # Let's assume mu1 = exp(X * beta_home) and mu2 = exp(X * beta_away).
        # But we only have one set of X (interaction features).
        
        # Let's try a simpler formulation first:
        # We model the difference directly using a linear predictor for the mean difference? No, Skellam is about counts.
        
        # Let's stick to the paper's likely approach or a standard one:
        # The paper likely used Skellam to model the goal difference distribution.
        # Let's assume mu1 (home goals) and mu2 (away goals) are constant or depend on team ratings.
        # Wait, if we use Skellam, we need two parameters mu1 and mu2 for each observation.
        # If we only have the goal difference as the target, we can't easily disentangle mu1 and mu2 without more constraints.
        
        # REVISED APPROACH:
        # Instead of a full custom Skellam regression class which might be unstable,
        # let's use the Skellam distribution to calculate probabilities *after* predicting expected goals.
        # We can continue to use Poisson/GLM to predict Home Goals and Away Goals separately (as we do now),
        # but use the Skellam distribution to calculate the probability of Win/Draw/Loss from those expected values.
        # This is a standard way to use Skellam in football: P(Home > Away) = sum(Skellam(k; mu1, mu2) for k > 0).
        
        # So, we don't necessarily need a new *training* model class if we predict mu1 and mu2 separately.
        # BUT, the paper says "Skellam Regression", implying they optimize the likelihood of the difference directly.
        # This is useful if we want to model the *net* effect of interactions.
        
        # Let's implement a helper to calculate match probabilities using Skellam,
        # which we can use in our existing pipeline or a new one.
        pass

    def fit(self, start_params=None, maxiter=1000, maxfun=5000, **kwargs):
        # Placeholder if we were to train it directly.
        return super(SkellamModel, self).fit(start_params, maxiter, maxfun, **kwargs)

def calculate_match_probs_skellam(mu_home, mu_away, max_diff=10):
    """
    Calculates Home Win, Draw, Away Win probabilities using Skellam distribution.
    
    Args:
        mu_home (float): Expected home goals.
        mu_away (float): Expected away goals.
        max_diff (int): Maximum goal difference to consider for summation (tail approximation).
        
    Returns:
        tuple: (prob_home_win, prob_draw, prob_away_win)
    """
    # P(Home Win) = P(Home - Away > 0) = P(Skellam > 0)
    # P(Draw) = P(Home - Away = 0) = P(Skellam = 0)
    # P(Away Win) = P(Home - Away < 0) = P(Skellam < 0)
    
    prob_draw = skellam.pmf(0, mu_home, mu_away)
    
    # Summing PMFs is accurate but can be slow if we do it for many matches.
    # CDF is better: P(X <= k).
    # P(Home Win) = P(X > 0) = 1 - P(X <= 0) = 1 - CDF(0)
    # P(Away Win) = P(X < 0) = P(X <= -1) = CDF(-1)
    
    prob_away_win = skellam.cdf(-1, mu_home, mu_away)
    prob_home_win = 1 - skellam.cdf(0, mu_home, mu_away)
    
    # Normalize to ensure sum is 1 (though Skellam is defined over all integers, so it should be close)
    total = prob_home_win + prob_draw + prob_away_win
    
    return prob_home_win / total, prob_draw / total, prob_away_win / total

if __name__ == "__main__":
    # Test
    h, d, a = calculate_match_probs_skellam(1.5, 1.2)
    print(f"Mu_Home=1.5, Mu_Away=1.2 -> H: {h:.4f}, D: {d:.4f}, A: {a:.4f}")
