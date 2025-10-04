"""
Retraining Orchestrator
Manages automated model retraining based on drift detection results
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..models.schemas import (
    DriftAlert,
    RetrainingJob,
    ModelVersion,
    RetrainingConfig,
    CostBenefitAnalysis
)
from .cost_analyzer import CostAnalyzer
from .model_manager import ModelManager
from ..utils.metrics import track_retraining_performance

logger = logging.getLogger(__name__)


@dataclass
class RetrainingDecision:
    """Result of retraining decision analysis"""
    should_retrain: bool
    confidence: float
    reason: str
    estimated_cost: float
    estimated_benefit: float
    priority: int
    recommended_config: Dict[str, Any]


class RetrainingOrchestrator:
    """
    Orchestrates automated model retraining based on drift detection
    and performance monitoring results
    """
    
    def __init__(
        self,
        cost_analyzer: CostAnalyzer,
        model_manager: ModelManager,
        config: Optional[RetrainingConfig] = None
    ):
        self.cost_analyzer = cost_analyzer
        self.model_manager = model_manager
        self.config = config or RetrainingConfig()
        
        # Retraining thresholds
        self.drift_threshold = self.config.drift_threshold
        self.performance_threshold = self.config.performance_threshold
        self.cost_benefit_ratio = self.config.cost_benefit_ratio
        
        # Retraining history
        self.retraining_history = []
        self.active_jobs = {}
        
        logger.info("Retraining orchestrator initialized")
    
    @track_retraining_performance
    async def evaluate_retraining_need(
        self,
        drift_alerts: List[DriftAlert],
        performance_metrics: Dict[str, float],
        model_id: str
    ) -> RetrainingDecision:
        """
        Evaluate whether a model needs retraining based on drift and performance
        
        Args:
            drift_alerts: List of drift detection alerts
            performance_metrics: Current model performance metrics
            model_id: Identifier for the model
            
        Returns:
            RetrainingDecision with recommendation
        """
        try:
            logger.info(f"Evaluating retraining need for model {model_id}")
            
            # Analyze drift severity
            drift_analysis = self._analyze_drift_severity(drift_alerts)
            
            # Analyze performance degradation
            performance_analysis = self._analyze_performance_degradation(
                performance_metrics, model_id
            )
            
            # Perform cost-benefit analysis
            cost_benefit = await self.cost_analyzer.analyze_retraining_cost_benefit(
                model_id, drift_analysis, performance_analysis
            )
            
            # Make retraining decision
            decision = self._make_retraining_decision(
                drift_analysis, performance_analysis, cost_benefit, model_id
            )
            
            logger.info(f"Retraining decision for {model_id}: {decision.should_retrain}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to evaluate retraining need: {str(e)}")
            return RetrainingDecision(
                should_retrain=False,
                confidence=0.0,
                reason=f"Evaluation failed: {str(e)}",
                estimated_cost=0.0,
                estimated_benefit=0.0,
                priority=0,
                recommended_config={}
            )
    
    async def execute_retraining(
        self,
        decision: RetrainingDecision,
        model_id: str,
        training_data: Optional[pd.DataFrame] = None
    ) -> RetrainingJob:
        """
        Execute model retraining based on decision
        
        Args:
            decision: Retraining decision
            model_id: Model identifier
            training_data: Optional new training data
            
        Returns:
            RetrainingJob with execution details
        """
        try:
            if not decision.should_retrain:
                raise ValueError("Retraining not recommended")
            
            logger.info(f"Starting retraining for model {model_id}")
            
            # Create retraining job
            job = RetrainingJob(
                job_id=UUID(),
                model_id=model_id,
                status="pending",
                priority=decision.priority,
                config=decision.recommended_config,
                estimated_cost=decision.estimated_cost,
                estimated_benefit=decision.estimated_benefit,
                created_at=datetime.utcnow()
            )
            
            # Add to active jobs
            self.active_jobs[str(job.job_id)] = job
            
            # Execute retraining
            await self._execute_retraining_pipeline(job, training_data)
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            
            # Add to history
            self.retraining_history.append(job)
            
            logger.info(f"Retraining completed for model {model_id}")
            
            return job
            
        except Exception as e:
            logger.error(f"Retraining execution failed: {str(e)}")
            if str(job.job_id) in self.active_jobs:
                self.active_jobs[str(job.job_id)].status = "failed"
                self.active_jobs[str(job.job_id)].error_message = str(e)
            raise
    
    def _analyze_drift_severity(self, drift_alerts: List[DriftAlert]) -> Dict[str, Any]:
        """Analyze the severity of drift alerts"""
        if not drift_alerts:
            return {"severity": "none", "confidence": 0.0, "affected_features": []}
        
        # Calculate overall severity
        severities = [alert.severity for alert in drift_alerts]
        severity_counts = {
            "high": severities.count("high"),
            "medium": severities.count("medium"),
            "low": severities.count("low")
        }
        
        # Determine overall severity
        if severity_counts["high"] > 0:
            overall_severity = "high"
        elif severity_counts["medium"] > 0:
            overall_severity = "medium"
        elif severity_counts["low"] > 0:
            overall_severity = "low"
        else:
            overall_severity = "none"
        
        # Calculate confidence
        confidences = [alert.confidence for alert in drift_alerts]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Get affected features
        affected_features = list(set([alert.feature_name for alert in drift_alerts]))
        
        return {
            "severity": overall_severity,
            "confidence": overall_confidence,
            "affected_features": affected_features,
            "alert_count": len(drift_alerts),
            "severity_distribution": severity_counts
        }
    
    def _analyze_performance_degradation(
        self, 
        performance_metrics: Dict[str, float], 
        model_id: str
    ) -> Dict[str, Any]:
        """Analyze model performance degradation"""
        try:
            # Get historical performance
            historical_metrics = self._get_historical_performance(model_id)
            
            if not historical_metrics:
                return {"degradation": "unknown", "confidence": 0.0, "metrics": {}}
            
            # Calculate performance changes
            performance_changes = {}
            for metric, current_value in performance_metrics.items():
                if metric in historical_metrics:
                    historical_value = historical_metrics[metric]
                    change = current_value - historical_value
                    change_percent = (change / historical_value) * 100
                    
                    performance_changes[metric] = {
                        "current": current_value,
                        "historical": historical_value,
                        "change": change,
                        "change_percent": change_percent
                    }
            
            # Determine overall degradation
            significant_degradation = any(
                abs(change["change_percent"]) > self.performance_threshold
                for change in performance_changes.values()
            )
            
            degradation_level = "high" if significant_degradation else "low"
            
            return {
                "degradation": degradation_level,
                "confidence": 0.8 if significant_degradation else 0.2,
                "metrics": performance_changes,
                "significant_changes": [
                    metric for metric, change in performance_changes.items()
                    if abs(change["change_percent"]) > self.performance_threshold
                ]
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return {"degradation": "unknown", "confidence": 0.0, "metrics": {}}
    
    def _make_retraining_decision(
        self,
        drift_analysis: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        cost_benefit: CostBenefitAnalysis,
        model_id: str
    ) -> RetrainingDecision:
        """Make final retraining decision based on all factors"""
        
        # Calculate decision score
        decision_score = 0.0
        reasons = []
        
        # Drift contribution
        drift_contribution = self._calculate_drift_contribution(drift_analysis)
        decision_score += drift_contribution["score"]
        if drift_contribution["score"] > 0:
            reasons.append(f"Drift detected: {drift_analysis['severity']} severity")
        
        # Performance contribution
        performance_contribution = self._calculate_performance_contribution(performance_analysis)
        decision_score += performance_contribution["score"]
        if performance_contribution["score"] > 0:
            reasons.append(f"Performance degradation: {performance_analysis['degradation']}")
        
        # Cost-benefit contribution
        cost_benefit_contribution = self._calculate_cost_benefit_contribution(cost_benefit)
        decision_score += cost_benefit_contribution["score"]
        if cost_benefit_contribution["score"] > 0:
            reasons.append(f"Cost-benefit ratio: {cost_benefit.ratio:.2f}")
        
        # Determine if retraining is recommended
        should_retrain = decision_score > self.drift_threshold
        
        # Calculate confidence
        confidence = min(1.0, decision_score)
        
        # Determine priority
        priority = self._calculate_priority(
            drift_analysis, performance_analysis, cost_benefit
        )
        
        # Get recommended configuration
        recommended_config = self._get_recommended_config(
            drift_analysis, performance_analysis, model_id
        )
        
        return RetrainingDecision(
            should_retrain=should_retrain,
            confidence=confidence,
            reason="; ".join(reasons) if reasons else "No significant issues detected",
            estimated_cost=cost_benefit.estimated_cost,
            estimated_benefit=cost_benefit.estimated_benefit,
            priority=priority,
            recommended_config=recommended_config
        )
    
    def _calculate_drift_contribution(self, drift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate contribution of drift analysis to decision score"""
        severity_scores = {"high": 0.8, "medium": 0.5, "low": 0.2, "none": 0.0}
        
        severity = drift_analysis.get("severity", "none")
        confidence = drift_analysis.get("confidence", 0.0)
        
        score = severity_scores.get(severity, 0.0) * confidence
        
        return {"score": score, "severity": severity, "confidence": confidence}
    
    def _calculate_performance_contribution(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate contribution of performance analysis to decision score"""
        degradation_scores = {"high": 0.7, "medium": 0.4, "low": 0.1, "unknown": 0.0}
        
        degradation = performance_analysis.get("degradation", "unknown")
        confidence = performance_analysis.get("confidence", 0.0)
        
        score = degradation_scores.get(degradation, 0.0) * confidence
        
        return {"score": score, "degradation": degradation, "confidence": confidence}
    
    def _calculate_cost_benefit_contribution(self, cost_benefit: CostBenefitAnalysis) -> Dict[str, Any]:
        """Calculate contribution of cost-benefit analysis to decision score"""
        if cost_benefit.ratio > self.cost_benefit_ratio:
            score = 0.5  # Positive contribution
        else:
            score = -0.2  # Negative contribution (costs outweigh benefits)
        
        return {"score": score, "ratio": cost_benefit.ratio}
    
    def _calculate_priority(
        self,
        drift_analysis: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        cost_benefit: CostBenefitAnalysis
    ) -> int:
        """Calculate retraining priority (1-10, higher is more urgent)"""
        priority = 5  # Base priority
        
        # Drift severity contribution
        severity_contributions = {"high": 3, "medium": 2, "low": 1, "none": 0}
        priority += severity_contributions.get(drift_analysis.get("severity", "none"), 0)
        
        # Performance degradation contribution
        degradation_contributions = {"high": 2, "medium": 1, "low": 0, "unknown": 0}
        priority += degradation_contributions.get(performance_analysis.get("degradation", "unknown"), 0)
        
        # Cost-benefit contribution
        if cost_benefit.ratio > 2.0:  # High benefit
            priority += 1
        elif cost_benefit.ratio < 0.5:  # Low benefit
            priority -= 1
        
        return max(1, min(10, priority))
    
    def _get_recommended_config(
        self,
        drift_analysis: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        model_id: str
    ) -> Dict[str, Any]:
        """Get recommended retraining configuration"""
        config = {
            "retraining_type": "full",
            "data_sampling": "stratified",
            "validation_split": 0.2,
            "hyperparameter_tuning": True,
            "ensemble_methods": False
        }
        
        # Adjust based on drift severity
        if drift_analysis.get("severity") == "high":
            config["retraining_type"] = "full"
            config["hyperparameter_tuning"] = True
            config["ensemble_methods"] = True
        
        # Adjust based on performance degradation
        if performance_analysis.get("degradation") == "high":
            config["validation_split"] = 0.3
            config["ensemble_methods"] = True
        
        return config
    
    async def _execute_retraining_pipeline(
        self, 
        job: RetrainingJob, 
        training_data: Optional[pd.DataFrame]
    ) -> None:
        """Execute the actual retraining pipeline"""
        try:
            # Update job status
            job.status = "running"
            job.started_at = datetime.utcnow()
            
            # Get model configuration
            model_config = await self.model_manager.get_model_config(job.model_id)
            
            # Prepare training data
            if training_data is None:
                training_data = await self.model_manager.get_training_data(job.model_id)
            
            # Execute retraining
            new_model = await self.model_manager.retrain_model(
                job.model_id, training_data, job.config
            )
            
            # Validate new model
            validation_results = await self.model_manager.validate_model(new_model)
            
            # Deploy new model if validation passes
            if validation_results["passed"]:
                await self.model_manager.deploy_model(new_model)
                job.deployed_model_version = new_model.version
            else:
                job.status = "failed"
                job.error_message = "Model validation failed"
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            raise
    
    def _get_historical_performance(self, model_id: str) -> Dict[str, float]:
        """Get historical performance metrics for a model"""
        # This would typically query a database
        # For now, return mock data
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        }
    
    async def get_retraining_status(self, model_id: str) -> Dict[str, Any]:
        """Get retraining status for a model"""
        active_jobs = [job for job in self.active_jobs.values() if job.model_id == model_id]
        historical_jobs = [job for job in self.retraining_history if job.model_id == model_id]
        
        return {
            "model_id": model_id,
            "active_jobs": len(active_jobs),
            "total_retrainings": len(historical_jobs),
            "last_retraining": historical_jobs[-1].created_at if historical_jobs else None,
            "next_evaluation": datetime.utcnow() + timedelta(hours=24)
        }

