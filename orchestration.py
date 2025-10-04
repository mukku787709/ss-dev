"""
Content Orchestration Service
Manages the multi-agent content generation pipeline
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import UUID

from ..models.schemas import (
    ContentRequest, 
    ContentResponse, 
    ProcessingStatus,
    AgentResult,
    AgentStatus
)
from ..agents.research_agent import ResearchAgent
from ..agents.outline_agent import OutlineAgent
from ..agents.writing_agent import WritingAgent
from ..agents.fact_check_agent import FactCheckAgent
from ..utils.metrics import track_orchestration_performance

logger = logging.getLogger(__name__)


class ContentOrchestrator:
    """
    Orchestrates the multi-agent content generation pipeline
    Manages agent execution, data flow, and quality gates
    """
    
    def __init__(
        self,
        openai_api_key: str,
        anthropic_api_key: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        
        # Initialize agents
        self.research_agent = ResearchAgent(anthropic_api_key)
        self.outline_agent = OutlineAgent(openai_api_key)
        self.writing_agent = WritingAgent(openai_api_key)
        self.fact_check_agent = FactCheckAgent(anthropic_api_key)
        
        # Pipeline configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.timeout = self.config.get("timeout", 300)  # 5 minutes
        self.quality_threshold = self.config.get("quality_threshold", 85.0)
        
        logger.info("Content orchestrator initialized with 4 specialized agents")
    
    @track_orchestration_performance
    async def process_content_request(
        self, 
        request: ContentRequest,
        content_id: UUID
    ) -> ContentResponse:
        """
        Process a content generation request through the multi-agent pipeline
        
        Args:
            request: Content generation request
            content_id: Unique identifier for the content request
            
        Returns:
            ContentResponse with final content and processing details
        """
        start_time = datetime.utcnow()
        agent_results = []
        total_cost = 0.0
        total_tokens = 0
        
        try:
            logger.info(f"Starting content processing for content_id: {content_id}")
            
            # Stage 1: Research Phase
            logger.info("Stage 1: Research Phase")
            research_result = await self._execute_research_phase(request)
            agent_results.append(research_result)
            total_cost += research_result.cost
            total_tokens += research_result.tokens_used
            
            if research_result.status == AgentStatus.FAILED:
                return self._create_failed_response(
                    content_id, agent_results, total_cost, total_tokens, start_time,
                    "Research phase failed"
                )
            
            # Stage 2: Outline Phase
            logger.info("Stage 2: Outline Phase")
            outline_result = await self._execute_outline_phase(request, research_result)
            agent_results.append(outline_result)
            total_cost += outline_result.cost
            total_tokens += outline_result.tokens_used
            
            if outline_result.status == AgentStatus.FAILED:
                return self._create_failed_response(
                    content_id, agent_results, total_cost, total_tokens, start_time,
                    "Outline phase failed"
                )
            
            # Stage 3: Writing Phase
            logger.info("Stage 3: Writing Phase")
            writing_result = await self._execute_writing_phase(request, research_result, outline_result)
            agent_results.append(writing_result)
            total_cost += writing_result.cost
            total_tokens += writing_result.tokens_used
            
            if writing_result.status == AgentStatus.FAILED:
                return self._create_failed_response(
                    content_id, agent_results, total_cost, total_tokens, start_time,
                    "Writing phase failed"
                )
            
            # Stage 4: Fact-Checking Phase
            logger.info("Stage 4: Fact-Checking Phase")
            fact_check_result = await self._execute_fact_check_phase(
                request, research_result, writing_result
            )
            agent_results.append(fact_check_result)
            total_cost += fact_check_result.cost
            total_tokens += fact_check_result.tokens_used
            
            # Stage 5: Quality Assessment
            logger.info("Stage 5: Quality Assessment")
            quality_score = await self._assess_content_quality(writing_result, fact_check_result)
            
            # Determine final status
            final_status = ProcessingStatus.COMPLETED
            if quality_score < self.quality_threshold:
                final_status = ProcessingStatus.REVIEWING
                logger.warning(f"Content quality below threshold: {quality_score} < {self.quality_threshold}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ContentResponse(
                content_id=content_id,
                status=final_status,
                final_content=writing_result.output,
                agent_results=agent_results,
                metadata={
                    "quality_score": quality_score,
                    "research_sources": research_result.metadata.get("sources_found", 0),
                    "fact_check_passed": fact_check_result.status == AgentStatus.COMPLETED,
                    "processing_pipeline": "multi_agent_v1"
                },
                total_cost=total_cost,
                total_tokens=total_tokens,
                processing_time=processing_time,
                quality_score=quality_score,
                created_at=start_time,
                updated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}", exc_info=True)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ContentResponse(
                content_id=content_id,
                status=ProcessingStatus.FAILED,
                agent_results=agent_results,
                metadata={"error": str(e)},
                total_cost=total_cost,
                total_tokens=total_tokens,
                processing_time=processing_time,
                created_at=start_time,
                updated_at=datetime.utcnow()
            )
    
    async def _execute_research_phase(self, request: ContentRequest) -> AgentResult:
        """Execute research phase with the research agent"""
        try:
            return await self.research_agent.research_topic(
                topic=request.prd_content,
                target_audience=request.target_audience,
                key_points=request.key_points,
                seo_keywords=request.seo_keywords
            )
        except Exception as e:
            logger.error(f"Research phase failed: {str(e)}")
            return AgentResult(
                agent_name="research_agent",
                status=AgentStatus.FAILED,
                error_message=str(e),
                processing_time=0.0,
                cost=0.0,
                tokens_used=0,
                timestamp=datetime.utcnow()
            )
    
    async def _execute_outline_phase(
        self, 
        request: ContentRequest, 
        research_result: AgentResult
    ) -> AgentResult:
        """Execute outline phase with the outline agent"""
        try:
            return await self.outline_agent.create_outline(
                prd_content=request.prd_content,
                research_data=research_result.output,
                content_type=request.content_type,
                target_audience=request.target_audience,
                seo_keywords=request.seo_keywords,
                word_count=request.word_count
            )
        except Exception as e:
            logger.error(f"Outline phase failed: {str(e)}")
            return AgentResult(
                agent_name="outline_agent",
                status=AgentStatus.FAILED,
                error_message=str(e),
                processing_time=0.0,
                cost=0.0,
                tokens_used=0,
                timestamp=datetime.utcnow()
            )
    
    async def _execute_writing_phase(
        self, 
        request: ContentRequest,
        research_result: AgentResult,
        outline_result: AgentResult
    ) -> AgentResult:
        """Execute writing phase with the writing agent"""
        try:
            return await self.writing_agent.write_content(
                outline=outline_result.output,
                research_data=research_result.output,
                content_type=request.content_type,
                brand_voice=request.brand_voice,
                target_audience=request.target_audience,
                word_count=request.word_count,
                custom_instructions=request.custom_instructions
            )
        except Exception as e:
            logger.error(f"Writing phase failed: {str(e)}")
            return AgentResult(
                agent_name="writing_agent",
                status=AgentStatus.FAILED,
                error_message=str(e),
                processing_time=0.0,
                cost=0.0,
                tokens_used=0,
                timestamp=datetime.utcnow()
            )
    
    async def _execute_fact_check_phase(
        self,
        request: ContentRequest,
        research_result: AgentResult,
        writing_result: AgentResult
    ) -> AgentResult:
        """Execute fact-checking phase with the fact-check agent"""
        try:
            return await self.fact_check_agent.verify_content(
                content=writing_result.output,
                research_sources=research_result.metadata.get("research_data", {}),
                include_citations=request.include_citations
            )
        except Exception as e:
            logger.error(f"Fact-check phase failed: {str(e)}")
            return AgentResult(
                agent_name="fact_check_agent",
                status=AgentStatus.FAILED,
                error_message=str(e),
                processing_time=0.0,
                cost=0.0,
                tokens_used=0,
                timestamp=datetime.utcnow()
            )
    
    async def _assess_content_quality(
        self, 
        writing_result: AgentResult, 
        fact_check_result: AgentResult
    ) -> float:
        """Assess overall content quality"""
        try:
            # Simple quality assessment based on agent results
            quality_factors = []
            
            # Writing quality (based on processing success)
            if writing_result.status == AgentStatus.COMPLETED:
                quality_factors.append(80.0)  # Base writing quality
            
            # Fact-checking quality
            if fact_check_result.status == AgentStatus.COMPLETED:
                quality_factors.append(90.0)  # Fact-checking passed
            else:
                quality_factors.append(60.0)  # Fact-checking failed
            
            # Calculate weighted average
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return 50.0  # Default quality score
    
    def _create_failed_response(
        self,
        content_id: UUID,
        agent_results: List[AgentResult],
        total_cost: float,
        total_tokens: int,
        start_time: datetime,
        error_message: str
    ) -> ContentResponse:
        """Create a failed response"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ContentResponse(
            content_id=content_id,
            status=ProcessingStatus.FAILED,
            agent_results=agent_results,
            metadata={"error": error_message},
            total_cost=total_cost,
            total_tokens=total_tokens,
            processing_time=processing_time,
            created_at=start_time,
            updated_at=datetime.utcnow()
        )
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics"""
        return {
            "pipeline_status": "operational",
            "agents": {
                "research_agent": await self.research_agent.get_agent_status(),
                "outline_agent": await self.outline_agent.get_agent_status(),
                "writing_agent": await self.writing_agent.get_agent_status(),
                "fact_check_agent": await self.fact_check_agent.get_agent_status()
            },
            "configuration": {
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "quality_threshold": self.quality_threshold
            },
            "last_updated": datetime.utcnow().isoformat()
        }

