"""
Pydantic schemas for Multi-Agent Content Operations
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Content processing status"""
    PENDING = "pending"
    RESEARCHING = "researching"
    OUTLINING = "outlining"
    WRITING = "writing"
    FACT_CHECKING = "fact_checking"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(str, Enum):
    """Individual agent status"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


class ContentType(str, Enum):
    """Content type enumeration"""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"


class BrandVoice(str, Enum):
    """Brand voice options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    AUTHORITATIVE = "authoritative"


class ContentRequest(BaseModel):
    """Request schema for content generation"""
    prd_content: str = Field(..., description="Product requirements document content")
    content_type: ContentType = Field(default=ContentType.BLOG_POST, description="Type of content to generate")
    brand_voice: BrandVoice = Field(default=BrandVoice.PROFESSIONAL, description="Brand voice for content")
    target_audience: str = Field(..., description="Target audience description")
    key_points: List[str] = Field(default=[], description="Key points to include")
    seo_keywords: List[str] = Field(default=[], description="SEO keywords to optimize for")
    word_count: int = Field(default=1000, ge=100, le=5000, description="Target word count")
    include_citations: bool = Field(default=True, description="Include citations and sources")
    custom_instructions: Optional[str] = Field(None, description="Custom instructions for content generation")
    
    @validator('word_count')
    def validate_word_count(cls, v):
        if v < 100 or v > 5000:
            raise ValueError('Word count must be between 100 and 5000')
        return v


class AgentResult(BaseModel):
    """Result from an individual agent"""
    agent_name: str
    status: AgentStatus
    output: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = Field(ge=0, description="Processing time in seconds")
    cost: float = Field(ge=0, description="Cost in USD")
    tokens_used: int = Field(ge=0, description="Number of tokens used")
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ContentResponse(BaseModel):
    """Response schema for content generation"""
    content_id: UUID = Field(default_factory=uuid4)
    status: ProcessingStatus
    final_content: Optional[str] = None
    agent_results: List[AgentResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    total_cost: float = Field(ge=0, description="Total cost in USD")
    total_tokens: int = Field(ge=0, description="Total tokens used")
    processing_time: float = Field(ge=0, description="Total processing time in seconds")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Content quality score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchData(BaseModel):
    """Research data structure"""
    sources: List[Dict[str, str]] = Field(default_factory=list)
    statistics: List[Dict[str, Any]] = Field(default_factory=list)
    trends: List[str] = Field(default_factory=list)
    expert_quotes: List[Dict[str, str]] = Field(default_factory=list)
    competitor_analysis: List[Dict[str, Any]] = Field(default_factory=list)


class OutlineStructure(BaseModel):
    """Content outline structure"""
    title: str
    introduction: str
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    conclusion: str
    call_to_action: str
    seo_meta: Dict[str, str] = Field(default_factory=dict)


class QualityMetrics(BaseModel):
    """Content quality metrics"""
    readability_score: float = Field(ge=0, le=100)
    engagement_score: float = Field(ge=0, le=100)
    seo_score: float = Field(ge=0, le=100)
    brand_consistency_score: float = Field(ge=0, le=100)
    fact_accuracy_score: float = Field(ge=0, le=100)
    overall_score: float = Field(ge=0, le=100)


class AgentConfig(BaseModel):
    """Configuration for individual agents"""
    model_name: str
    temperature: float = Field(ge=0, le=2, default=0.7)
    max_tokens: int = Field(ge=100, le=4000, default=2000)
    timeout: int = Field(ge=30, le=300, default=120)
    retry_count: int = Field(ge=0, le=5, default=3)
    cost_limit: float = Field(ge=0, le=100, default=10.0)


class ProcessingConfig(BaseModel):
    """Global processing configuration"""
    research_agent: AgentConfig
    outline_agent: AgentConfig
    writing_agent: AgentConfig
    fact_check_agent: AgentConfig
    max_total_cost: float = Field(ge=0, le=1000, default=50.0)
    max_processing_time: int = Field(ge=60, le=3600, default=300)


class ErrorResponse(BaseModel):
    """Error response schema"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchRequest(BaseModel):
    """Batch processing request"""
    requests: List[ContentRequest] = Field(..., min_items=1, max_items=100)
    priority: int = Field(default=5, ge=1, le=10)
    callback_url: Optional[str] = None


class BatchResponse(BaseModel):
    """Batch processing response"""
    batch_id: UUID = Field(default_factory=uuid4)
    total_requests: int
    queued_requests: int
    estimated_completion_time: datetime
    status: ProcessingStatus

