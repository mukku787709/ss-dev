"""
Intelligent Model Router
Routes requests to optimal models based on complexity, cost, and performance
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..models.schemas import (
    RoutingRequest,
    RoutingResponse,
    ModelSelection,
    CostAnalysis,
    QualityMetrics
)
from .complexity_analyzer import ComplexityAnalyzer
from .model_selector import ModelSelector
from .quality_monitor import QualityMonitor
from ..cost_optimization.cost_calculator import CostCalculator
from ..cost_optimization.budget_manager import BudgetManager
from ..utils.metrics import track_routing_performance

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of routing decision analysis"""
    selected_model: str
    confidence: float
    estimated_cost: float
    estimated_quality: float
    reasoning: str
    fallback_models: List[str]
    routing_time: float


class IntelligentRouter:
    """
    Intelligent model router that selects optimal models based on
    complexity analysis, cost optimization, and quality requirements
    """
    
    def __init__(
        self,
        complexity_analyzer: ComplexityAnalyzer,
        model_selector: ModelSelector,
        quality_monitor: QualityMonitor,
        cost_calculator: CostCalculator,
        budget_manager: BudgetManager
    ):
        self.complexity_analyzer = complexity_analyzer
        self.model_selector = model_selector
        self.quality_monitor = quality_monitor
        self.cost_calculator = cost_calculator
        self.budget_manager = budget_manager
        
        # Routing configuration
        self.max_routing_time = 50  # milliseconds
        self.quality_threshold = 0.95
        self.cost_optimization_weight = 0.7
        self.quality_weight = 0.3
        
        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "average_routing_time": 0.0,
            "cost_savings": 0.0
        }
        
        logger.info("Intelligent model router initialized")
    
    @track_routing_performance
    async def route_request(
        self,
        request: RoutingRequest,
        user_id: Optional[str] = None
    ) -> RoutingResponse:
        """
        Route a request to the optimal model based on complexity and constraints
        
        Args:
            request: Routing request with content and constraints
            user_id: Optional user identifier for budget tracking
            
        Returns:
            RoutingResponse with selected model and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Routing request for user {user_id}")
            
            # Step 1: Analyze request complexity
            complexity_analysis = await self._analyze_complexity(request)
            
            # Step 2: Get available models and their capabilities
            available_models = await self._get_available_models(request)
            
            # Step 3: Calculate costs for each model
            cost_analysis = await self._calculate_costs(request, available_models)
            
            # Step 4: Check budget constraints
            budget_constraints = await self._check_budget_constraints(
                cost_analysis, user_id
            )
            
            # Step 5: Select optimal model
            routing_decision = await self._select_optimal_model(
                request, complexity_analysis, available_models,
                cost_analysis, budget_constraints
            )
            
            # Step 6: Execute request with selected model
            response = await self._execute_request(request, routing_decision)
            
            # Step 7: Update performance metrics
            routing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            await self._update_routing_metrics(routing_decision, routing_time)
            
            logger.info(f"Request routed to {routing_decision.selected_model} in {routing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Routing failed: {str(e)}", exc_info=True)
            # Fallback to default model
            return await self._fallback_route(request, str(e))
    
    async def _analyze_complexity(self, request: RoutingRequest) -> Dict[str, Any]:
        """Analyze the complexity of the request"""
        try:
            complexity_score = await self.complexity_analyzer.analyze_complexity(
                request.content,
                request.task_type,
                request.requirements
            )
            
            return {
                "score": complexity_score,
                "category": self._categorize_complexity(complexity_score),
                "features": await self.complexity_analyzer.extract_features(request.content)
            }
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {str(e)}")
            return {
                "score": 0.5,  # Default medium complexity
                "category": "medium",
                "features": {}
            }
    
    def _categorize_complexity(self, score: float) -> str:
        """Categorize complexity score into discrete categories"""
        if score < 0.3:
            return "low"
        elif score < 0.7:
            return "medium"
        else:
            return "high"
    
    async def _get_available_models(self, request: RoutingRequest) -> List[Dict[str, Any]]:
        """Get available models that can handle the request"""
        try:
            # Get all available models
            all_models = await self.model_selector.get_available_models()
            
            # Filter models based on request requirements
            suitable_models = []
            for model in all_models:
                if self._is_model_suitable(model, request):
                    suitable_models.append(model)
            
            return suitable_models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    def _is_model_suitable(self, model: Dict[str, Any], request: RoutingRequest) -> bool:
        """Check if a model is suitable for the request"""
        # Check task type compatibility
        if request.task_type not in model.get("supported_tasks", []):
            return False
        
        # Check quality requirements
        if request.min_quality and model.get("quality_score", 0) < request.min_quality:
            return False
        
        # Check model availability
        if not model.get("available", True):
            return False
        
        return True
    
    async def _calculate_costs(
        self, 
        request: RoutingRequest, 
        models: List[Dict[str, Any]]
    ) -> Dict[str, CostAnalysis]:
        """Calculate costs for each model"""
        cost_analysis = {}
        
        for model in models:
            try:
                cost = await self.cost_calculator.calculate_cost(
                    model["id"],
                    request.content,
                    model["pricing"]
                )
                
                cost_analysis[model["id"]] = CostAnalysis(
                    model_id=model["id"],
                    estimated_cost=cost,
                    cost_per_token=model["pricing"]["cost_per_token"],
                    cost_per_request=model["pricing"]["cost_per_request"],
                    total_tokens=request.estimated_tokens or 1000
                )
                
            except Exception as e:
                logger.error(f"Cost calculation failed for model {model['id']}: {str(e)}")
                cost_analysis[model["id"]] = CostAnalysis(
                    model_id=model["id"],
                    estimated_cost=float('inf'),
                    cost_per_token=0.0,
                    cost_per_request=0.0,
                    total_tokens=0
                )
        
        return cost_analysis
    
    async def _check_budget_constraints(
        self, 
        cost_analysis: Dict[str, CostAnalysis], 
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Check budget constraints for the user"""
        if not user_id:
            return {"budget_available": True, "remaining_budget": float('inf')}
        
        try:
            budget_info = await self.budget_manager.get_budget_info(user_id)
            remaining_budget = budget_info["remaining_budget"]
            
            # Check if any model is within budget
            affordable_models = [
                model_id for model_id, cost in cost_analysis.items()
                if cost.estimated_cost <= remaining_budget
            ]
            
            return {
                "budget_available": len(affordable_models) > 0,
                "remaining_budget": remaining_budget,
                "affordable_models": affordable_models
            }
            
        except Exception as e:
            logger.error(f"Budget check failed: {str(e)}")
            return {"budget_available": True, "remaining_budget": float('inf')}
    
    async def _select_optimal_model(
        self,
        request: RoutingRequest,
        complexity_analysis: Dict[str, Any],
        available_models: List[Dict[str, Any]],
        cost_analysis: Dict[str, CostAnalysis],
        budget_constraints: Dict[str, Any]
    ) -> RoutingDecision:
        """Select the optimal model based on all factors"""
        
        # Filter models based on budget constraints
        if not budget_constraints["budget_available"]:
            available_models = [
                model for model in available_models
                if model["id"] in budget_constraints.get("affordable_models", [])
            ]
        
        if not available_models:
            raise ValueError("No suitable models available within budget")
        
        # Calculate scores for each model
        model_scores = []
        for model in available_models:
            score = await self._calculate_model_score(
                model, complexity_analysis, cost_analysis, request
            )
            model_scores.append((model, score))
        
        # Sort by score (higher is better)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best model
        best_model, best_score = model_scores[0]
        
        # Get fallback models
        fallback_models = [model["id"] for model, score in model_scores[1:3]]
        
        # Calculate confidence
        confidence = min(1.0, best_score / 100.0)
        
        return RoutingDecision(
            selected_model=best_model["id"],
            confidence=confidence,
            estimated_cost=cost_analysis[best_model["id"]].estimated_cost,
            estimated_quality=best_model.get("quality_score", 0.8),
            reasoning=f"Selected based on complexity {complexity_analysis['category']} and cost optimization",
            fallback_models=fallback_models,
            routing_time=0.0  # Will be set by caller
        )
    
    async def _calculate_model_score(
        self,
        model: Dict[str, Any],
        complexity_analysis: Dict[str, Any],
        cost_analysis: Dict[str, CostAnalysis],
        request: RoutingRequest
    ) -> float:
        """Calculate a score for model selection"""
        
        # Base score from model quality
        quality_score = model.get("quality_score", 0.8) * 40
        
        # Cost optimization score (lower cost is better)
        cost = cost_analysis[model["id"]].estimated_cost
        max_cost = max(cost_analysis.values(), key=lambda x: x.estimated_cost).estimated_cost
        cost_score = (1 - cost / max_cost) * 30
        
        # Complexity matching score
        complexity_score = self._calculate_complexity_match_score(
            model, complexity_analysis
        ) * 20
        
        # Performance score
        performance_score = model.get("performance_score", 0.8) * 10
        
        total_score = quality_score + cost_score + complexity_score + performance_score
        
        return total_score
    
    def _calculate_complexity_match_score(
        self, 
        model: Dict[str, Any], 
        complexity_analysis: Dict[str, Any]
    ) -> float:
        """Calculate how well the model matches the complexity requirements"""
        
        model_complexity = model.get("complexity_handling", "medium")
        request_complexity = complexity_analysis["category"]
        
        # Perfect match
        if model_complexity == request_complexity:
            return 1.0
        
        # Over-provisioning (expensive but acceptable)
        if (model_complexity == "high" and request_complexity in ["low", "medium"]) or \
           (model_complexity == "medium" and request_complexity == "low"):
            return 0.7
        
        # Under-provisioning (risky)
        if (model_complexity == "low" and request_complexity in ["medium", "high"]) or \
           (model_complexity == "medium" and request_complexity == "high"):
            return 0.3
        
        return 0.5  # Default score
    
    async def _execute_request(
        self, 
        request: RoutingRequest, 
        routing_decision: RoutingDecision
    ) -> RoutingResponse:
        """Execute the request with the selected model"""
        
        try:
            # Get model provider
            model_provider = await self.model_selector.get_model_provider(
                routing_decision.selected_model
            )
            
            # Execute request
            start_time = time.time()
            result = await model_provider.process_request(request)
            execution_time = (time.time() - start_time) * 1000
            
            # Monitor quality
            quality_metrics = await self.quality_monitor.assess_quality(
                request, result, routing_decision.selected_model
            )
            
            # Update budget
            if request.user_id:
                await self.budget_manager.update_usage(
                    request.user_id, routing_decision.estimated_cost
                )
            
            return RoutingResponse(
                request_id=request.request_id,
                selected_model=routing_decision.selected_model,
                response=result,
                cost=routing_decision.estimated_cost,
                quality_metrics=quality_metrics,
                execution_time=execution_time,
                routing_time=routing_decision.routing_time,
                confidence=routing_decision.confidence,
                fallback_models=routing_decision.fallback_models,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Request execution failed: {str(e)}")
            # Try fallback models
            for fallback_model in routing_decision.fallback_models:
                try:
                    model_provider = await self.model_selector.get_model_provider(fallback_model)
                    result = await model_provider.process_request(request)
                    
                    return RoutingResponse(
                        request_id=request.request_id,
                        selected_model=fallback_model,
                        response=result,
                        cost=0.0,  # Will be calculated
                        quality_metrics={},
                        execution_time=0.0,
                        routing_time=routing_decision.routing_time,
                        confidence=0.5,
                        fallback_models=[],
                        timestamp=datetime.utcnow()
                    )
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model {fallback_model} also failed: {str(fallback_error)}")
                    continue
            
            # All models failed
            raise Exception("All models failed to process request")
    
    async def _fallback_route(self, request: RoutingRequest, error: str) -> RoutingResponse:
        """Fallback routing when all else fails"""
        
        # Use default model
        default_model = "gpt-3.5-turbo"  # Default fallback
        
        try:
            model_provider = await self.model_selector.get_model_provider(default_model)
            result = await model_provider.process_request(request)
            
            return RoutingResponse(
                request_id=request.request_id,
                selected_model=default_model,
                response=result,
                cost=0.0,
                quality_metrics={},
                execution_time=0.0,
                routing_time=0.0,
                confidence=0.3,
                fallback_models=[],
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Fallback routing also failed: {str(e)}")
            raise Exception(f"All routing options failed: {error}")
    
    async def _update_routing_metrics(
        self, 
        routing_decision: RoutingDecision, 
        routing_time: float
    ) -> None:
        """Update routing performance metrics"""
        
        self.routing_stats["total_requests"] += 1
        self.routing_stats["successful_routes"] += 1
        
        # Update average routing time
        current_avg = self.routing_stats["average_routing_time"]
        total_requests = self.routing_stats["total_requests"]
        self.routing_stats["average_routing_time"] = (
            (current_avg * (total_requests - 1) + routing_time) / total_requests
        )
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics"""
        return {
            **self.routing_stats,
            "success_rate": (
                self.routing_stats["successful_routes"] / 
                max(1, self.routing_stats["total_requests"])
            ),
            "average_cost_savings": self.routing_stats["cost_savings"] / 
            max(1, self.routing_stats["total_requests"])
        }

