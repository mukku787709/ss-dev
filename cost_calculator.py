"""
Cost Calculator
Calculates and optimizes costs for different AI models and providers
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..models.schemas import CostAnalysis, PricingModel, UsageMetrics
from ..utils.metrics import track_cost_calculation

logger = logging.getLogger(__name__)


@dataclass
class CostOptimizationResult:
    """Result of cost optimization analysis"""
    optimal_model: str
    cost_savings: float
    quality_impact: float
    recommendation: str
    alternative_models: List[Dict[str, Any]]


class CostCalculator:
    """
    Calculates and optimizes costs for AI model usage
    Supports multiple pricing models and cost optimization strategies
    """
    
    def __init__(self):
        self.pricing_models = self._initialize_pricing_models()
        self.cost_history = {}
        self.optimization_cache = {}
        
        logger.info("Cost calculator initialized")
    
    def _initialize_pricing_models(self) -> Dict[str, PricingModel]:
        """Initialize pricing models for different providers"""
        return {
            # OpenAI Models
            "gpt-4": PricingModel(
                provider="openai",
                model_name="gpt-4",
                cost_per_token=0.00003,  # $0.03 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=8192,
                quality_score=0.95
            ),
            "gpt-4-turbo": PricingModel(
                provider="openai",
                model_name="gpt-4-turbo",
                cost_per_token=0.00001,  # $0.01 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=128000,
                quality_score=0.93
            ),
            "gpt-3.5-turbo": PricingModel(
                provider="openai",
                model_name="gpt-3.5-turbo",
                cost_per_token=0.000002,  # $0.002 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=4096,
                quality_score=0.85
            ),
            
            # Anthropic Models
            "claude-3-opus": PricingModel(
                provider="anthropic",
                model_name="claude-3-opus",
                cost_per_token=0.000015,  # $0.015 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=200000,
                quality_score=0.96
            ),
            "claude-3-sonnet": PricingModel(
                provider="anthropic",
                model_name="claude-3-sonnet",
                cost_per_token=0.000003,  # $0.003 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=200000,
                quality_score=0.92
            ),
            "claude-3-haiku": PricingModel(
                provider="anthropic",
                model_name="claude-3-haiku",
                cost_per_token=0.00000025,  # $0.00025 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=200000,
                quality_score=0.88
            ),
            
            # Cohere Models
            "command": PricingModel(
                provider="cohere",
                model_name="command",
                cost_per_token=0.0000015,  # $0.0015 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=4096,
                quality_score=0.82
            ),
            "command-light": PricingModel(
                provider="cohere",
                model_name="command-light",
                cost_per_token=0.0000005,  # $0.0005 per 1K tokens
                cost_per_request=0.0,
                min_tokens=1,
                max_tokens=4096,
                quality_score=0.78
            )
        }
    
    @track_cost_calculation
    async def calculate_cost(
        self,
        model_id: str,
        content: str,
        pricing: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the cost for using a specific model
        
        Args:
            model_id: Identifier for the model
            content: Content to process
            pricing: Optional custom pricing model
            
        Returns:
            Estimated cost in USD
        """
        try:
            # Get pricing model
            if pricing:
                pricing_model = PricingModel(**pricing)
            else:
                pricing_model = self.pricing_models.get(model_id)
            
            if not pricing_model:
                logger.warning(f"No pricing model found for {model_id}")
                return 0.0
            
            # Estimate token count
            estimated_tokens = self._estimate_tokens(content)
            
            # Calculate cost
            token_cost = estimated_tokens * pricing_model.cost_per_token
            request_cost = pricing_model.cost_per_request
            
            total_cost = token_cost + request_cost
            
            # Cache the calculation
            self.cost_history[model_id] = {
                "tokens": estimated_tokens,
                "cost": total_cost,
                "timestamp": datetime.utcnow()
            }
            
            logger.debug(f"Cost calculated for {model_id}: ${total_cost:.4f}")
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Cost calculation failed for {model_id}: {str(e)}")
            return 0.0
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        # Simple estimation: ~4 characters per token
        # In production, use proper tokenization
        return max(1, len(content) // 4)
    
    async def calculate_batch_cost(
        self,
        requests: List[Dict[str, Any]],
        model_id: str
    ) -> Dict[str, Any]:
        """
        Calculate cost for a batch of requests
        
        Args:
            requests: List of request data
            model_id: Model to use for all requests
            
        Returns:
            Batch cost analysis
        """
        try:
            total_cost = 0.0
            individual_costs = []
            
            for request in requests:
                cost = await self.calculate_cost(
                    model_id, 
                    request.get("content", ""),
                    request.get("pricing")
                )
                total_cost += cost
                individual_costs.append(cost)
            
            return {
                "total_cost": total_cost,
                "average_cost": total_cost / len(requests),
                "individual_costs": individual_costs,
                "request_count": len(requests),
                "model_id": model_id
            }
            
        except Exception as e:
            logger.error(f"Batch cost calculation failed: {str(e)}")
            return {
                "total_cost": 0.0,
                "average_cost": 0.0,
                "individual_costs": [],
                "request_count": 0,
                "model_id": model_id
            }
    
    async def optimize_costs(
        self,
        requests: List[Dict[str, Any]],
        quality_requirements: Dict[str, float],
        budget_constraints: Dict[str, Any]
    ) -> CostOptimizationResult:
        """
        Optimize costs while meeting quality requirements
        
        Args:
            requests: List of requests to optimize
            quality_requirements: Minimum quality requirements
            budget_constraints: Budget limits and constraints
            
        Returns:
            Cost optimization result
        """
        try:
            # Analyze request complexity
            complexity_analysis = await self._analyze_request_complexity(requests)
            
            # Find suitable models
            suitable_models = self._find_suitable_models(
                quality_requirements, budget_constraints
            )
            
            # Calculate costs for each model
            model_costs = {}
            for model_id in suitable_models:
                batch_cost = await self.calculate_batch_cost(requests, model_id)
                model_costs[model_id] = batch_cost
            
            # Find optimal model
            optimal_model = self._find_optimal_model(
                model_costs, quality_requirements, budget_constraints
            )
            
            # Calculate savings
            baseline_cost = min(model_costs.values(), key=lambda x: x["total_cost"])["total_cost"]
            optimal_cost = model_costs[optimal_model]["total_cost"]
            cost_savings = baseline_cost - optimal_cost
            
            # Get alternative models
            alternative_models = self._get_alternative_models(
                model_costs, optimal_model, quality_requirements
            )
            
            return CostOptimizationResult(
                optimal_model=optimal_model,
                cost_savings=cost_savings,
                quality_impact=0.0,  # Will be calculated based on actual usage
                recommendation=f"Use {optimal_model} for {cost_savings:.2f} cost savings",
                alternative_models=alternative_models
            )
            
        except Exception as e:
            logger.error(f"Cost optimization failed: {str(e)}")
            return CostOptimizationResult(
                optimal_model="gpt-3.5-turbo",  # Default fallback
                cost_savings=0.0,
                quality_impact=0.0,
                recommendation="Fallback to default model",
                alternative_models=[]
            )
    
    async def _analyze_request_complexity(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the complexity of requests"""
        total_length = sum(len(req.get("content", "")) for req in requests)
        average_length = total_length / len(requests) if requests else 0
        
        # Categorize complexity
        if average_length < 100:
            complexity = "low"
        elif average_length < 1000:
            complexity = "medium"
        else:
            complexity = "high"
        
        return {
            "complexity": complexity,
            "average_length": average_length,
            "total_requests": len(requests),
            "total_length": total_length
        }
    
    def _find_suitable_models(
        self,
        quality_requirements: Dict[str, float],
        budget_constraints: Dict[str, Any]
    ) -> List[str]:
        """Find models that meet quality and budget requirements"""
        suitable_models = []
        
        for model_id, pricing in self.pricing_models.items():
            # Check quality requirements
            if pricing.quality_score >= quality_requirements.get("min_quality", 0.8):
                # Check budget constraints
                if pricing.cost_per_token <= budget_constraints.get("max_cost_per_token", float('inf')):
                    suitable_models.append(model_id)
        
        return suitable_models
    
    def _find_optimal_model(
        self,
        model_costs: Dict[str, Dict[str, Any]],
        quality_requirements: Dict[str, float],
        budget_constraints: Dict[str, Any]
    ) -> str:
        """Find the optimal model based on cost and quality"""
        
        # Score each model
        model_scores = {}
        for model_id, cost_info in model_costs.items():
            pricing = self.pricing_models[model_id]
            
            # Cost score (lower cost is better)
            cost_score = 1.0 / (cost_info["total_cost"] + 0.001)
            
            # Quality score
            quality_score = pricing.quality_score
            
            # Combined score
            total_score = cost_score * 0.6 + quality_score * 0.4
            
            model_scores[model_id] = total_score
        
        # Return model with highest score
        return max(model_scores, key=model_scores.get)
    
    def _get_alternative_models(
        self,
        model_costs: Dict[str, Dict[str, Any]],
        optimal_model: str,
        quality_requirements: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Get alternative models with their trade-offs"""
        alternatives = []
        
        for model_id, cost_info in model_costs.items():
            if model_id != optimal_model:
                pricing = self.pricing_models[model_id]
                
                alternatives.append({
                    "model_id": model_id,
                    "cost": cost_info["total_cost"],
                    "quality": pricing.quality_score,
                    "cost_difference": cost_info["total_cost"] - model_costs[optimal_model]["total_cost"],
                    "quality_difference": pricing.quality_score - self.pricing_models[optimal_model].quality_score
                })
        
        # Sort by cost
        alternatives.sort(key=lambda x: x["cost"])
        
        return alternatives[:3]  # Return top 3 alternatives
    
    async def get_cost_analytics(
        self,
        time_period: str = "7d",
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost analytics for a time period"""
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            if time_period == "1d":
                start_time = end_time - timedelta(days=1)
            elif time_period == "7d":
                start_time = end_time - timedelta(days=7)
            elif time_period == "30d":
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(days=7)
            
            # Filter cost history
            filtered_costs = {
                model: data for model, data in self.cost_history.items()
                if data["timestamp"] >= start_time and (not model_id or model == model_id)
            }
            
            if not filtered_costs:
                return {
                    "total_cost": 0.0,
                    "average_cost": 0.0,
                    "model_breakdown": {},
                    "time_period": time_period
                }
            
            # Calculate analytics
            total_cost = sum(data["cost"] for data in filtered_costs.values())
            average_cost = total_cost / len(filtered_costs)
            
            # Model breakdown
            model_breakdown = {}
            for model, data in filtered_costs.items():
                if model not in model_breakdown:
                    model_breakdown[model] = {"cost": 0.0, "requests": 0}
                model_breakdown[model]["cost"] += data["cost"]
                model_breakdown[model]["requests"] += 1
            
            return {
                "total_cost": total_cost,
                "average_cost": average_cost,
                "model_breakdown": model_breakdown,
                "time_period": time_period,
                "request_count": len(filtered_costs)
            }
            
        except Exception as e:
            logger.error(f"Cost analytics failed: {str(e)}")
            return {
                "total_cost": 0.0,
                "average_cost": 0.0,
                "model_breakdown": {},
                "time_period": time_period,
                "request_count": 0
            }
    
    async def predict_costs(
        self,
        estimated_requests: int,
        average_request_size: int,
        model_id: str
    ) -> Dict[str, Any]:
        """Predict costs for future usage"""
        try:
            pricing = self.pricing_models.get(model_id)
            if not pricing:
                return {"error": f"No pricing model found for {model_id}"}
            
            # Calculate predicted costs
            estimated_tokens = estimated_requests * (average_request_size // 4)
            predicted_cost = estimated_tokens * pricing.cost_per_token
            
            return {
                "model_id": model_id,
                "estimated_requests": estimated_requests,
                "estimated_tokens": estimated_tokens,
                "predicted_cost": predicted_cost,
                "cost_per_request": predicted_cost / estimated_requests if estimated_requests > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Cost prediction failed: {str(e)}")
            return {"error": str(e)}

