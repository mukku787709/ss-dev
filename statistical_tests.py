"""
Statistical Drift Detection Tests
Implements various statistical tests for detecting data and concept drift
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class StatisticalDriftDetector:
    """
    Statistical drift detection using various statistical tests
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.test_results = {}
        
    def detect_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray,
        feature_name: str = "feature"
    ) -> Dict[str, any]:
        """
        Detect drift between reference and current data using multiple tests
        
        Args:
            reference_data: Reference dataset (baseline)
            current_data: Current dataset to compare
            feature_name: Name of the feature being tested
            
        Returns:
            Dictionary with drift detection results
        """
        results = {
            "feature_name": feature_name,
            "drift_detected": False,
            "confidence": 0.0,
            "tests": {},
            "severity": "none"
        }
        
        try:
            # Kolmogorov-Smirnov Test (for continuous features)
            ks_result = self._ks_test(reference_data, current_data)
            results["tests"]["ks_test"] = ks_result
            
            # Population Stability Index (PSI)
            psi_result = self._psi_test(reference_data, current_data)
            results["tests"]["psi"] = psi_result
            
            # Mann-Whitney U Test (non-parametric)
            mw_result = self._mann_whitney_test(reference_data, current_data)
            results["tests"]["mann_whitney"] = mw_result
            
            # Effect Size (Cohen's d)
            effect_size = self._cohens_d(reference_data, current_data)
            results["tests"]["effect_size"] = effect_size
            
            # Determine overall drift
            drift_detected, confidence, severity = self._determine_drift(results["tests"])
            results["drift_detected"] = drift_detected
            results["confidence"] = confidence
            results["severity"] = severity
            
            logger.info(f"Drift detection completed for {feature_name}: {drift_detected}")
            
        except Exception as e:
            logger.error(f"Drift detection failed for {feature_name}: {str(e)}")
            results["error"] = str(e)
            
        return results
    
    def _ks_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, any]:
        """Kolmogorov-Smirnov test for distribution comparison"""
        try:
            statistic, p_value = ks_2samp(reference, current)
            return {
                "test_name": "Kolmogorov-Smirnov",
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "interpretation": "Distribution difference" if p_value < self.significance_level else "No significant difference"
            }
        except Exception as e:
            logger.error(f"KS test failed: {str(e)}")
            return {"error": str(e)}
    
    def _psi_test(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> Dict[str, any]:
        """Population Stability Index test"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Calculate PSI
            psi = self._calculate_psi(reference, current, bin_edges)
            
            # Interpret PSI value
            if psi < 0.1:
                interpretation = "No significant change"
            elif psi < 0.2:
                interpretation = "Slight change"
            else:
                interpretation = "Significant change"
            
            return {
                "test_name": "Population Stability Index",
                "psi_value": psi,
                "significant": psi > 0.2,
                "interpretation": interpretation
            }
        except Exception as e:
            logger.error(f"PSI test failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bin_edges: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        # Get bin counts for reference and current data
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to probabilities
        ref_probs = ref_counts / len(reference)
        curr_probs = curr_counts / len(current)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        ref_probs = np.where(ref_probs == 0, epsilon, ref_probs)
        curr_probs = np.where(curr_probs == 0, epsilon, curr_probs)
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        return psi
    
    def _mann_whitney_test(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, any]:
        """Mann-Whitney U test (non-parametric)"""
        try:
            statistic, p_value = mannwhitneyu(reference, current, alternative='two-sided')
            return {
                "test_name": "Mann-Whitney U",
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "interpretation": "Distribution difference" if p_value < self.significance_level else "No significant difference"
            }
        except Exception as e:
            logger.error(f"Mann-Whitney test failed: {str(e)}")
            return {"error": str(e)}
    
    def _cohens_d(self, reference: np.ndarray, current: np.ndarray) -> Dict[str, any]:
        """Calculate Cohen's d effect size"""
        try:
            # Calculate means and standard deviations
            ref_mean = np.mean(reference)
            curr_mean = np.mean(current)
            ref_std = np.std(reference, ddof=1)
            curr_std = np.std(current, ddof=1)
            
            # Pooled standard deviation
            n1, n2 = len(reference), len(current)
            pooled_std = np.sqrt(((n1 - 1) * ref_std**2 + (n2 - 1) * curr_std**2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohens_d = (curr_mean - ref_mean) / pooled_std
            
            # Interpret effect size
            if abs(cohens_d) < 0.2:
                interpretation = "Negligible effect"
            elif abs(cohens_d) < 0.5:
                interpretation = "Small effect"
            elif abs(cohens_d) < 0.8:
                interpretation = "Medium effect"
            else:
                interpretation = "Large effect"
            
            return {
                "test_name": "Cohen's d",
                "effect_size": cohens_d,
                "interpretation": interpretation,
                "significant": abs(cohens_d) > 0.5
            }
        except Exception as e:
            logger.error(f"Cohen's d calculation failed: {str(e)}")
            return {"error": str(e)}
    
    def _determine_drift(
        self, 
        test_results: Dict[str, Dict[str, any]]
    ) -> Tuple[bool, float, str]:
        """
        Determine overall drift based on multiple test results
        
        Args:
            test_results: Results from multiple statistical tests
            
        Returns:
            Tuple of (drift_detected, confidence, severity)
        """
        significant_tests = 0
        total_tests = 0
        confidence_scores = []
        
        for test_name, result in test_results.items():
            if "error" not in result:
                total_tests += 1
                if result.get("significant", False):
                    significant_tests += 1
                    confidence_scores.append(0.8)  # High confidence for significant tests
                else:
                    confidence_scores.append(0.2)  # Low confidence for non-significant tests
        
        if total_tests == 0:
            return False, 0.0, "none"
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Determine drift based on majority of tests
        drift_detected = significant_tests > (total_tests / 2)
        
        # Determine severity
        if significant_tests >= total_tests * 0.8:
            severity = "high"
        elif significant_tests >= total_tests * 0.5:
            severity = "medium"
        elif significant_tests > 0:
            severity = "low"
        else:
            severity = "none"
        
        return drift_detected, overall_confidence, severity
    
    def detect_categorical_drift(
        self, 
        reference_data: List[str], 
        current_data: List[str]
    ) -> Dict[str, any]:
        """
        Detect drift in categorical features using chi-square test
        
        Args:
            reference_data: Reference categorical data
            current_data: Current categorical data
            
        Returns:
            Dictionary with drift detection results
        """
        try:
            # Create contingency table
            ref_counts = pd.Series(reference_data).value_counts()
            curr_counts = pd.Series(current_data).value_counts()
            
            # Get all unique categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            # Create contingency table
            contingency_table = []
            for category in all_categories:
                ref_count = ref_counts.get(category, 0)
                curr_count = curr_counts.get(category, 0)
                contingency_table.append([ref_count, curr_count])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            return {
                "test_name": "Chi-square",
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant": p_value < self.significance_level,
                "interpretation": "Distribution difference" if p_value < self.significance_level else "No significant difference"
            }
            
        except Exception as e:
            logger.error(f"Categorical drift detection failed: {str(e)}")
            return {"error": str(e)}
    
    def detect_multivariate_drift(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray
    ) -> Dict[str, any]:
        """
        Detect multivariate drift using Maximum Mean Discrepancy (MMD)
        
        Args:
            reference_data: Reference multivariate data
            current_data: Current multivariate data
            
        Returns:
            Dictionary with drift detection results
        """
        try:
            # Calculate MMD using RBF kernel
            mmd_statistic = self._calculate_mmd(reference_data, current_data)
            
            # Bootstrap test for significance
            p_value = self._bootstrap_mmd_test(reference_data, current_data, n_bootstrap=1000)
            
            return {
                "test_name": "Maximum Mean Discrepancy",
                "mmd_statistic": mmd_statistic,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "interpretation": "Multivariate distribution difference" if p_value < self.significance_level else "No significant difference"
            }
            
        except Exception as e:
            logger.error(f"Multivariate drift detection failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Calculate Maximum Mean Discrepancy using RBF kernel"""
        # RBF kernel function
        def rbf_kernel(X, Y, gamma):
            pairwise_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :] - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * pairwise_dists)
        
        # Calculate MMD^2
        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)
        
        mmd_squared = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
        
        return np.sqrt(max(0, mmd_squared))
    
    def _bootstrap_mmd_test(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        n_bootstrap: int = 1000
    ) -> float:
        """Bootstrap test for MMD significance"""
        # Combine data
        combined_data = np.vstack([X, Y])
        n_x, n_y = len(X), len(Y)
        
        # Calculate original MMD
        original_mmd = self._calculate_mmd(X, Y)
        
        # Bootstrap samples
        mmd_bootstrap = []
        for _ in range(n_bootstrap):
            # Randomly shuffle combined data
            np.random.shuffle(combined_data)
            
            # Split into two groups
            X_boot = combined_data[:n_x]
            Y_boot = combined_data[n_x:]
            
            # Calculate MMD for bootstrap sample
            mmd_boot = self._calculate_mmd(X_boot, Y_boot)
            mmd_bootstrap.append(mmd_boot)
        
        # Calculate p-value
        p_value = np.mean(np.array(mmd_bootstrap) >= original_mmd)
        
        return p_value

