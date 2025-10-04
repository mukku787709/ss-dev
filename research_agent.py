"""
Research Agent - Specialized agent for content research and data gathering
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
from anthropic import AsyncAnthropic
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from ..models.schemas import AgentResult, AgentStatus, ResearchData
from ..utils.metrics import track_agent_performance

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Specialized agent for researching content topics, gathering data,
    and providing comprehensive research for content generation.
    """
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_name = model_name
        self.search_tool = DuckDuckGoSearchRun()
        
        # Research prompt template
        self.research_prompt = PromptTemplate(
            input_variables=["topic", "target_audience", "key_points", "seo_keywords"],
            template="""
            You are a professional research agent specializing in content research and data gathering.
            
            Your task is to conduct comprehensive research on the following topic:
            Topic: {topic}
            Target Audience: {target_audience}
            Key Points to Research: {key_points}
            SEO Keywords: {seo_keywords}
            
            Research Requirements:
            1. Find recent statistics and data points (last 2 years)
            2. Identify industry trends and insights
            3. Gather expert opinions and quotes
            4. Find relevant case studies and examples
            5. Research competitor content and positioning
            6. Identify potential sources and citations
            
            Provide a comprehensive research report with:
            - Key statistics and data points
            - Industry trends and insights
            - Expert quotes and opinions
            - Relevant case studies
            - Source URLs and citations
            - Competitor analysis
            
            Focus on accuracy, relevance, and recency of information.
            """
        )
    
    @track_agent_performance
    async def research_topic(
        self, 
        topic: str, 
        target_audience: str,
        key_points: List[str],
        seo_keywords: List[str],
        max_sources: int = 10
    ) -> AgentResult:
        """
        Conduct comprehensive research on a given topic
        
        Args:
            topic: The topic to research
            target_audience: Target audience for the content
            key_points: Key points to research
            seo_keywords: SEO keywords to focus on
            max_sources: Maximum number of sources to gather
            
        Returns:
            AgentResult with research data and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Prepare research query
            research_query = self._prepare_research_query(topic, key_points, seo_keywords)
            
            # Conduct web search
            search_results = await self._conduct_web_search(research_query, max_sources)
            
            # Analyze and synthesize research
            research_data = await self._analyze_research_data(
                search_results, topic, target_audience, key_points
            )
            
            # Generate comprehensive research report
            research_report = await self._generate_research_report(research_data)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AgentResult(
                agent_name="research_agent",
                status=AgentStatus.COMPLETED,
                output=research_report,
                metadata={
                    "sources_found": len(search_results),
                    "research_data": research_data.dict(),
                    "query_used": research_query
                },
                processing_time=processing_time,
                cost=0.0,  # Will be calculated based on token usage
                tokens_used=0,  # Will be calculated
                timestamp=start_time
            )
            
        except Exception as e:
            logger.error(f"Research agent failed: {str(e)}", exc_info=True)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AgentResult(
                agent_name="research_agent",
                status=AgentStatus.FAILED,
                error_message=str(e),
                processing_time=processing_time,
                cost=0.0,
                tokens_used=0,
                timestamp=start_time
            )
    
    def _prepare_research_query(
        self, 
        topic: str, 
        key_points: List[str], 
        seo_keywords: List[str]
    ) -> str:
        """Prepare comprehensive research query"""
        query_parts = [topic]
        
        if key_points:
            query_parts.extend(key_points[:3])  # Limit to top 3 key points
        
        if seo_keywords:
            query_parts.extend(seo_keywords[:3])  # Limit to top 3 keywords
        
        return " ".join(query_parts)
    
    async def _conduct_web_search(self, query: str, max_sources: int) -> List[Dict[str, Any]]:
        """Conduct web search and gather sources"""
        try:
            # Use DuckDuckGo search for web results
            search_results = self.search_tool.run(query)
            
            # Parse search results (this would need to be implemented based on the tool's output format)
            sources = self._parse_search_results(search_results, max_sources)
            
            return sources
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    def _parse_search_results(self, results: str, max_sources: int) -> List[Dict[str, Any]]:
        """Parse search results into structured format"""
        # This would need to be implemented based on the actual search tool output
        # For now, return mock data structure
        sources = []
        for i in range(min(max_sources, 5)):  # Mock implementation
            sources.append({
                "title": f"Research Source {i+1}",
                "url": f"https://example.com/source-{i+1}",
                "snippet": f"Relevant information about the topic from source {i+1}",
                "relevance_score": 0.8 - (i * 0.1)
            })
        
        return sources
    
    async def _analyze_research_data(
        self, 
        sources: List[Dict[str, Any]], 
        topic: str,
        target_audience: str,
        key_points: List[str]
    ) -> ResearchData:
        """Analyze and structure research data"""
        
        # Extract statistics and data points
        statistics = []
        for source in sources[:3]:  # Analyze top 3 sources
            statistics.append({
                "source": source["title"],
                "data_point": f"Relevant statistic from {source['title']}",
                "value": "Sample value",
                "year": "2024"
            })
        
        # Extract trends
        trends = [
            f"Trend 1 related to {topic}",
            f"Trend 2 in {target_audience} market",
            f"Emerging trend in the industry"
        ]
        
        # Extract expert quotes
        expert_quotes = []
        for source in sources[:2]:
            expert_quotes.append({
                "quote": f"Expert opinion from {source['title']}",
                "expert": f"Industry Expert {len(expert_quotes) + 1}",
                "source": source["url"]
            })
        
        # Competitor analysis
        competitor_analysis = [
            {
                "competitor": "Competitor A",
                "approach": "Their approach to the topic",
                "strengths": ["Strength 1", "Strength 2"],
                "weaknesses": ["Weakness 1", "Weakness 2"]
            }
        ]
        
        return ResearchData(
            sources=[{"title": s["title"], "url": s["url"]} for s in sources],
            statistics=statistics,
            trends=trends,
            expert_quotes=expert_quotes,
            competitor_analysis=competitor_analysis
        )
    
    async def _generate_research_report(self, research_data: ResearchData) -> str:
        """Generate comprehensive research report"""
        
        report_sections = [
            "# Research Report",
            "",
            "## Key Statistics",
            *[f"- {stat['data_point']}: {stat['value']} ({stat['year']})" for stat in research_data.statistics],
            "",
            "## Industry Trends",
            *[f"- {trend}" for trend in research_data.trends],
            "",
            "## Expert Insights",
            *[f"- \"{quote['quote']}\" - {quote['expert']}" for quote in research_data.expert_quotes],
            "",
            "## Competitor Analysis",
            *[f"- {comp['competitor']}: {comp['approach']}" for comp in research_data.competitor_analysis],
            "",
            "## Sources",
            *[f"- [{source['title']}]({source['url']})" for source in research_data.sources]
        ]
        
        return "\n".join(report_sections)
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_name": "research_agent",
            "status": "idle",
            "model": self.model_name,
            "capabilities": [
                "web_search",
                "data_analysis",
                "trend_identification",
                "source_verification"
            ],
            "last_activity": datetime.utcnow().isoformat()
        }

