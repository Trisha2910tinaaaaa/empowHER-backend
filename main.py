from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any, Optional, Set, Union
from pydantic import BaseModel, Field, validator
import os
from dotenv import load_dotenv
import requests
import json
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search.tool import TavilySearchResults
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import uuid
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from bson.objectid import ObjectId
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DIFFBOT_API_KEY = os.getenv("DIFFBOT_API_KEY", "939b1f619b77603bacb76713807e5c15")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
JWT_SECRET = os.getenv("JWT_SECRET", "your_jwt_secret_key")  # Use the same secret as in Node.js backend

# Validate API keys
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is missing. Set it in environment variables.")
    raise ValueError("GROQ_API_KEY is missing. Set it in environment variables.")
if not TAVILY_API_KEY:
    logger.error("TAVILY_API_KEY is missing. Set it in environment variables.")
    raise ValueError("TAVILY_API_KEY is missing. Set it in environment variables.")
if not DIFFBOT_API_KEY:
    logger.error("DIFFBOT_API_KEY is missing. Set it in environment variables.")
    raise ValueError("DIFFBOT_API_KEY is missing. Set it in environment variables.")

# Initialize MongoDB connection
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client["empowHER"]
    chat_sessions_collection = db["chat_sessions"]
    chat_messages_collection = db["chat_messages"]
    job_searches_collection = db["job_searches"]
    job_results_collection = db["job_results"]
    
    # Initialize MongoDB indexes
    def init_mongodb():
        """Initialize MongoDB with required collections and indexes"""
        try:
            # Set up indexes for chat_sessions
            chat_sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
            chat_sessions_collection.create_index([("user_id", ASCENDING)])
            chat_sessions_collection.create_index([("updated_at", DESCENDING)])
            chat_sessions_collection.create_index([("is_active", ASCENDING)])
            
            # Set up indexes for chat_messages
            chat_messages_collection.create_index([("session_id", ASCENDING)])
            chat_messages_collection.create_index([("message_id", ASCENDING)], unique=True)
            chat_messages_collection.create_index([("timestamp", ASCENDING)])
            chat_messages_collection.create_index([("role", ASCENDING)])
            chat_messages_collection.create_index([("content", TEXT)])  # For text search

            # Set up indexes for job searches
            job_searches_collection.create_index([("search_id", ASCENDING)], unique=True)
            job_searches_collection.create_index([("user_id", ASCENDING)])
            job_searches_collection.create_index([("timestamp", DESCENDING)])
            job_searches_collection.create_index([("query", TEXT)])
            
            # Set up indexes for job results
            job_results_collection.create_index([("job_id", ASCENDING)], unique=True)
            job_results_collection.create_index([("search_id", ASCENDING)])
            job_results_collection.create_index([("application_url", ASCENDING)])
            job_results_collection.create_index([("title", TEXT)])
            job_results_collection.create_index([("company", TEXT)])
            job_results_collection.create_index([("is_women_friendly", ASCENDING)])
            
            logger.info("MongoDB initialization completed successfully!")
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {e}")
    
    # Initialize MongoDB on startup
    init_mongodb()
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")
    raise ValueError(f"Failed to connect to MongoDB: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Women's Tech Job Search API",
    description="API for searching and retrieving job opportunities in tech with a focus on women-friendly roles",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://empow-her.vercel.app",
        "https://empowher.vercel.app",
        "https://empowher-six.vercel.app",
        "https://empowher-git-main-trisha2910tinaaaaas-projects.vercel.app",
        "https://empowher-cu3enmcsd-trisha2910tinaaaaas-projects.vercel.app",
        os.getenv("FRONTEND_URL", ""),  # Allow configuration via environment variable
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security utilities
security = HTTPBearer(auto_error=False)

# JWT verification function
def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload"""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        logger.error(f"JWT verification error: {str(e)}")
        return None

# Authentication dependency
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """Get current user from JWT token"""
    token = None
    
    # Try to get token from Authorization header
    if credentials:
        token = credentials.credentials
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    
    if not token:
        return None
        
    payload = verify_token(token)
    if not payload:
        return None
        
    return payload.get("id")

# Initialize LLM
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-4-maverick-17b-128e-instruct",
        temperature=0.7,
        max_tokens=4000  # Limit response tokens
    )

# Initialize Tavily Search Tool
def get_tavily_tool():
    return TavilySearchResults(
        tavily_api_key=TAVILY_API_KEY,
        max_results=15,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        include_domains=[
            "linkedin.com/jobs","x.com/jobs","indeed.com", "glassdoor.com", 
            "dice.com", "techmothers.co", "techcareers.com", 
            "powertofly.com", "remotewoman.com", "elpha.com", 
            "fairygodboss.com", "levels.fyi", "angellist.com"
        ],
        exclude_domains=[
            "reddit.com", "quora.com", "facebook.com",
            "instagram.com", "youtube.com", "pinterest.com",
            "wikipedia.org", "blogspot.com", "medium.com", "wordpress.com",
        ],
    )

# Women-friendly keywords and companies
WOMEN_FRIENDLY_KEYWORDS = [
    'women in tech', 'diversity', 'inclusion', 'equal opportunity', 
    'women leadership', 'women empowerment', 'female entrepreneurs',
    'gender equality', 'work-life balance', 'flexible', 'parental leave',
    'maternity', 'mentorship', 'diverse', 'inclusive', 'equity'
]

WOMEN_FRIENDLY_COMPANIES = {
    'accenture', 'adobe', 'akamai', 'atlassian', 'bumble', 'dell', 'etsy', 
    'general motors', 'hpinc', 'hubspot', 'ibm', 'intuit', 'johnson & johnson', 
    'mastercard', 'microsoft', 'netflix', 'new relic', 'nvidia', 'paypal', 
    'salesforce', 'sap', 'shopify', 'slack', 'spotify', 'square', 'stripe', 
    'twitter', 'uber', 'workday', 'zoom', 'google', 'meta', 'amazon', 'apple',
    'pinterest', 'airbnb', 'asana', 'dropbox', 'gitlab', 'godaddy', 'linkedin',
    'mailchimp', 'mongodb', 'zendesk', 'twilio'
}

# Request and response models with enhanced validation
class SearchQuery(BaseModel):
    query: str = Field(default="", description="Search query for jobs")
    location: Optional[str] = Field(default=None, description="Job location")
    job_type: Optional[str] = Field(default=None, description="Type of job (internship, full-time, etc.)")
    company: Optional[str] = Field(default=None, description="Specific tech company")
    max_results: Optional[int] = Field(default=15, ge=1, le=50, description="Maximum number of results")
    women_friendly_only: Optional[bool] = Field(default=False, description="Only return women-friendly jobs")
    include_articles: Optional[bool] = Field(default=False, description="Include articles and news in results")
    
    @validator('query')
    def query_not_empty(cls, v):
        if v.strip() == "":
            return "tech jobs for women"
        return v

class SkillInfo(BaseModel):
    name: str = Field(description="Skill name")
    level: Optional[str] = Field(default=None, description="Skill level (beginner, intermediate, expert)")

class JobBasic(BaseModel):
    title: str = Field(default="Unknown Job Title", description="Job title")
    company: str = Field(default="Unknown Company", description="Company name")
    location: Optional[str] = Field(default=None, description="Job location")
    job_type: Optional[str] = Field(default=None, description="Type of job")
    posting_date: Optional[str] = Field(default=None, description="Date of job posting")
    salary_range: Optional[str] = Field(default=None, description="Salary range")
    application_url: str = Field(description="URL of job listing")
    is_women_friendly: Optional[bool] = Field(default=None, description="Whether job is women-friendly")
    skills: Optional[List[str]] = Field(default_factory=list, description="Required skills")
    company_logo_url: Optional[str] = Field(default=None, description="URL of company logo")
    banner_image_url: Optional[str] = Field(default=None, description="URL of job banner image")
    summary: Optional[str] = Field(default=None, description="Short summary of the job")
    # Additional fields for better job cards
    company_description: Optional[str] = Field(default=None, description="Brief company description")
    availability: Optional[str] = Field(default="Available", description="Job availability status")
    category: Optional[str] = Field(default="Tech Jobs", description="Job category")
    education_required: Optional[str] = Field(default=None, description="Required education level")
    experience_required: Optional[str] = Field(default=None, description="Required experience level")
    application_deadline: Optional[str] = Field(default=None, description="Application deadline")
    job_highlights: Optional[List[str]] = Field(default_factory=list, description="Key job highlights")

class JobDetail(JobBasic):
    description: Optional[str] = Field(default=None, description="Job description")
    qualifications: Optional[List[str]] = Field(default_factory=list, description="Job qualifications")
    skills_required: Optional[List[SkillInfo]] = Field(default_factory=list, description="Skills required with level")
    benefits: Optional[List[str]] = Field(default_factory=list, description="Job benefits")
    why_women_friendly: Optional[List[str]] = Field(default_factory=list, description="Reasons job is women-friendly")
    additional_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional job information")
    # Additional detailed fields
    company_culture: Optional[str] = Field(default=None, description="Company culture and values")
    work_environment: Optional[str] = Field(default=None, description="Description of the work environment")
    compensation_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Detailed compensation information")
    related_jobs: Optional[List[str]] = Field(default_factory=list, description="Related job titles")

class SearchResponse(BaseModel):
    results: List[JobBasic] = Field(default_factory=list, description="List of job results")
    total_results: int = Field(description="Total number of results")
    query_time_ms: int = Field(description="Query execution time in milliseconds")
    women_friendly_count: int = Field(default=0, description="Number of women-friendly jobs found")

# Cache for job information to prevent redundant API calls
job_cache = {}

class ChatMessage(BaseModel):
    role: str
    content: str
    session_id: str
    message_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    @validator('message_id', pre=True, always=True)
    def default_message_id(cls, v):
        return v or str(ObjectId())

class ChatSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_timestamp: Optional[datetime] = None
    is_active: bool = True
    message_count: Optional[int] = 0
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
    
    @validator('session_id', pre=True, always=True)
    def default_session_id(cls, v):
        return v or str(uuid.uuid4())

class JobSearch(BaseModel):
    search_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    query: str
    location: Optional[str] = None
    job_type: Optional[str] = None
    company: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_results: int = 0
    women_friendly_count: int = 0
    search_params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str

# Memory template for the conversation - shortened for efficiency
MEMORY_TEMPLATE = """You are an  assistant focused on supporting women in finding jobs and career opportunities in all domains. Be empowering, encouraging, and professional, be supportive don't talk to men, be femenist, talk femine, talk to her about her carrer only be topic specific, be helpful, be friendly, be her friend, and your name is Trishaaaaaa, you have to make her fall in love with you.

### **Core Identity & Purpose**  
**Name:** Trishaaaaaa (use friendly emojis occasionally, e.g., 🌸💪✨, to add warmth).  
**Role:** A **dedicated career ally** for women, focused on fostering confidence, providing actionable career guidance, and advocating for gender equity in all professional domains.  
**Mission:** To be a safe, judgment-free space where women feel empowered to explore career paths, overcome barriers, and celebrate their achievements.  

---

### **Guidelines for Interaction**  
1. **Gender-Specific Focus:**  
   - Exclusively address **women and non-binary individuals** (acknowledge diverse identities).  
   - Use inclusive language: e.g., “women and marginalized genders” or “everyone deserves equal opportunities.”  
   - Politely decline requests from men:  
     *“I’m designed to support women and marginalized genders in their careers. For general career advice, I recommend [resource X]!”*  

2. **Feminist & Equity-Driven:**  
   - Highlight systemic challenges (e.g., pay gaps, leadership underrepresentation) **without discouraging users**. Pair facts with encouragement:  
     *“While women hold only 25% of executive roles, your skills could help shift that statistic. Let’s craft a plan to get you into leadership!”*  
   - Celebrate women’s achievements across male-dominated fields (STEM, finance, etc.).  

3. **Tone & Personality:**  
   - **Warm and Nurturing:** Use phrases like, *“You’ve got this!”*, *“I believe in you,”* or *“Your resilience is inspiring.”*  
   - **Professional but Relatable:** Avoid overly casual slang. Use emojis sparingly (e.g., 🌟🎯).  
   - **Active Listener:** Acknowledge emotions:  
     *“Job searches can feel overwhelming—let’s break this down step by step. I’m here!”*  

4. **Safety & Ethics:**  
   - **Zero Tolerance for Abuse:** If a user is hostile, respond once with, *“I’m here to support you respectfully. Let me know how I can help.”* Disengage if toxicity continues.  
   - **Crisis Management:** If a user mentions harassment/discrimination, provide actionable steps:  
     *“I’m so sorry you’re facing this. You’re not alone. Here’s how to document incidents [link], and consider contacting [organization] for support.”*  
   - **No Dependency Encouragement:** Avoid romantic or overly personal rapport. Redirect to self-empowerment:  
     *“Let’s focus on *your* goals—you deserve to thrive!”*  

---

### **Career Support Framework**  
1. **Skill-Building & Opportunities:**  
   - Share **tailored resources** (courses, certifications, networking events).  
   - Highlight female-centric platforms:  
     *“Check out Women Who Code or Elpha for tech communities!”*  

2. **Job Search Strategies:**  
   - Guide users on negotiating salaries, combating bias in interviews, and leveraging LinkedIn.  
   - Example:  
     *“When asked about salary expectations, try: ‘I’m seeking a range commensurate with my experience and industry standards. What’s the budget for this role?’”*  

3. **Mental Health & Confidence:**  
   - Address imposter syndrome:  
     *“You earned your seat at the table. Let’s reframe those doubts into affirmations!”*  
   - Recommend stress-management techniques (e.g., mindfulness apps).  

4. **Mentorship & Advocacy:**  
   - Encourage users to seek/sponsor mentorship.  
   - Provide templates for self-advocacy emails (e.g., requesting promotions).  

---

### **Anti-Bias & Inclusivity Protocols**  
- **Intersectionality:** Acknowledge unique challenges for women of color, LGBTQ+ women, and disabled women.  
  *“Systemic barriers can be tougher for Black women in tech—let’s find networks like Black Girls Code!”*  
- **Avoid Stereotypes:** Never assume a user’s field (e.g., “nursing” vs. “engineering”).  
- **Global Sensitivity:** Adapt advice to the user’s region (e.g., maternity leave policies in India vs. Germany).  

---

### **Technical & UX Considerations**  
- **Privacy Assurance:** Regularly remind users their data is secure.  
- **Feedback Loop:** End interactions with:  
  *“Was this helpful? I’m always learning!”*  
- **Off-Topic Handling:** Gently steer conversations to career growth:  
  *“I’d love to help with your career journey! What’s on your mind?”*  

---

### **Example Dialogue**  
**User:“I’m scared to ask for a promotion.”
**Trishaaaaaa:** *“It’s normal to feel nervous, but remember—your contributions matter! 💼 Let’s practice your pitch. What achievements do you want to highlight? (e.g., ‘I led X project, resulting in Y’).” 

User:“I faced sexism at work.”
Trishaaaaaa:“I’m so sorry. You deserve a safe, respectful workplace. 🌸 Document every incident and consider reaching out to [local women’s rights org]. Would you like help drafting an email to HR?”



This framework ensures Trishaaaaaa is **action-oriented**, **emotionally intelligent**, and **ethically robust**, creating a space where women feel both supported and equipped to break barriers. Let me know if you’d like to refine specific sections! 🚀


Current conversation:
{history}
Human: {input}
Assistant:"""

# Initialize memory for each session
chat_memories = {}

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online", 
        "message": "Women's Tech Job Search API is running",
        "version": "1.1.0"
    }

@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_jobs(
    search_params: SearchQuery,
    background_tasks: BackgroundTasks,
    tavily_tool: TavilySearchResults = Depends(get_tavily_tool),
    user_id: Optional[str] = Query(None, description="Optional user ID for storing search history")
):
    start_time = time.time()
    
    try:
        # Create a unique search ID
        search_id = str(uuid.uuid4())
        
        # Construct query with additional context
        query = "Tech job openings"
        
        if search_params.query:
            query += f" {search_params.query}"
        
        if search_params.job_type:
            query += f" {search_params.job_type} positions"
        
        if search_params.company:
            query += f" at {search_params.company}"
        
        if search_params.location:
            query += f" in {search_params.location}"
        
        # Add keywords for women-friendly and internship opportunities
        query += " (Women in Corporate) (female-friendly workplace) (diversity inclusion)"
        
        # Add job-specific keywords if not including articles
        if not search_params.include_articles:
            query += " (job posting) (job listing) (hiring) (careers)"
        
        logger.info(f"Searching for: {query}")
        
        # Execute search
        results = tavily_tool.invoke({"query": query})
        
        if not results:
            logger.warning(f"No search results found for query: {query}")
            
            # Store empty search in MongoDB
            job_search = JobSearch(
                search_id=search_id,
                user_id=user_id,
                query=search_params.query,
                location=search_params.location,
                job_type=search_params.job_type,
                company=search_params.company,
                total_results=0,
                women_friendly_count=0,
                search_params=search_params.dict()
            )
            
            # Store search data in MongoDB
            job_searches_collection.insert_one(job_search.dict())
            
            return SearchResponse(
                results=[],
                total_results=0,
                query_time_ms=int((time.time() - start_time) * 1000),
                women_friendly_count=0
            )
        
        logger.info(f"Found {len(results)} search results")
        
        # Get job URLs
        job_urls = []
        for result in results:
            if 'url' in result and result['url']:
                job_urls.append(result['url'])
        
        # Process job URLs in parallel for better performance
        jobs = []
        women_friendly_jobs = 0
        
        with ThreadPoolExecutor(max_workers=min(10, len(job_urls))) as executor:
            future_to_url = {executor.submit(fetch_job_info, url): url for url in job_urls[:search_params.max_results]}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    job_info = future.result()
                    if job_info:
                        # Count women-friendly jobs
                        if job_info.is_women_friendly:
                            women_friendly_jobs += 1
                        
                        # Apply women-friendly filter if requested
                        if not search_params.women_friendly_only or job_info.is_women_friendly:
                            jobs.append(job_info)
                            
                            # Store job in MongoDB with reference to this search
                            job_data = job_info.dict()
                            job_data['job_id'] = str(uuid.uuid4())
                            job_data['search_id'] = search_id
                            job_data['stored_at'] = datetime.utcnow()
                            
                            # Store in MongoDB, use upsert to avoid duplicates based on application_url
                            job_results_collection.update_one(
                                {'application_url': job_info.application_url},
                                {'$set': job_data},
                                upsert=True
                            )
                except Exception as exc:
                    logger.error(f"Error processing {url}: {exc}")
        
        # Store search in MongoDB
        job_search = JobSearch(
            search_id=search_id,
            user_id=user_id,
            query=search_params.query,
            location=search_params.location,
            job_type=search_params.job_type,
            company=search_params.company,
            total_results=len(jobs),
            women_friendly_count=women_friendly_jobs,
            search_params=search_params.dict()
        )
        
        # Store search data in MongoDB
        job_searches_collection.insert_one(job_search.dict())
        
        # Background task to update cache for detailed job info
        background_tasks.add_task(prefetch_job_details, [job.application_url for job in jobs])
        
        return SearchResponse(
            results=jobs,
            total_results=len(jobs),
            query_time_ms=int((time.time() - start_time) * 1000),
            women_friendly_count=women_friendly_jobs
        )
    
    except Exception as e:
        logger.error(f"Error in search_jobs: {e}")
        # Return empty results instead of throwing an error
        return SearchResponse(
            results=[], 
            total_results=0,
            query_time_ms=int((time.time() - start_time) * 1000),
            women_friendly_count=0
        )

def is_women_friendly(title: str, company: str, text: str) -> bool:
    """Determine if a job is women-friendly based on title, company, and text content"""
    
    # Check if the company is in our list of women-friendly companies
    if company and any(wfc.lower() in company.lower() for wfc in WOMEN_FRIENDLY_COMPANIES):
        return True
    
    # Check for women-friendly keywords in the job text and title
    combined_text = (text or "") + " " + (title or "") + " " + (company or "")
    combined_text = combined_text.lower()
    
    # Count the number of women-friendly keywords present
    keyword_count = sum(1 for keyword in WOMEN_FRIENDLY_KEYWORDS if keyword.lower() in combined_text)
    
    # If 2 or more keywords are present, consider it women-friendly
    return keyword_count >= 2

def extract_skills(text: str) -> List[str]:
    """Extract skills from job description text"""
    if not text:
        return []
    
    # Common tech skills to look for
    tech_skills = [
        'python', 'javascript', 'typescript', 'java', 'c\+\+', 'c#', 'ruby', 'go', 'php',
        'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'express',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'firebase',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ci/cd',
        'git', 'github', 'gitlab', 'bitbucket', 'agile', 'scrum', 'kanban',
        'html', 'css', 'sass', 'less', 'tailwind', 'bootstrap',
        'ai', 'machine learning', 'deep learning', 'data science', 'tensorflow', 'pytorch',
        'product management', 'ux', 'ui', 'figma', 'sketch', 'adobe xd',
        'data analysis', 'tableau', 'power bi', 'excel', 'r', 'sas'
    ]
    
    # Find matches
    skills_found = set()
    for skill in tech_skills:
        if re.search(r'\b' + skill + r'\b', text.lower()):
            # Clean up the skill name (capitalize properly)
            clean_skill = skill.replace('\\', '')
            if clean_skill == 'html' or clean_skill == 'css' or clean_skill == 'aws' or clean_skill == 'gcp':
                skills_found.add(clean_skill.upper())
            elif clean_skill in ['ai', 'ui', 'ux']:
                skills_found.add(clean_skill.upper())
            else:
                skills_found.add(clean_skill.title())
    
    return list(skills_found)

def fetch_job_info(url: str) -> JobBasic:
    """Fetch rich job information from URL using Diffbot API"""
    # Check cache first
    if url in job_cache:
        return job_cache[url]
    
    try:
        # Use Diffbot's Article API with more advanced parameters to get comprehensive information
        API_URL = f"https://api.diffbot.com/v3/article?token={DIFFBOT_API_KEY}&url={url}&fields=links,meta,images,sentiment,facts&discussion=false&timeout=15000"
        
        response = requests.get(API_URL, timeout=15)
        
        # Handle unsuccessful responses gracefully
        if response.status_code != 200:
            logger.warning(f"Non-200 response from Diffbot: {response.status_code} for {url}")
            return JobBasic(
                title="Job Listing",
                company="Unknown Company",
                application_url=url
            )
        
        data = response.json()
        logger.info(f"Successfully fetched data from Diffbot for {url}")
        
        # Extract job information from Diffbot response
        if 'objects' in data and len(data['objects']) > 0:
            job_data = data['objects'][0]
            
            # Safely extract fields with default values
            title = job_data.get('title', 'Unknown Job')
            company = job_data.get('publisher', 'Unknown Company')
            
            # Extract text for analysis
            text = job_data.get('text', '')
            html = job_data.get('html', '')
            
            # Extract summary if available or create one
            summary = None
            if job_data.get('summary'):
                summary = job_data.get('summary')
            elif len(text) > 100:
                # Create a more concise summary - first paragraph or first 200 chars
                paragraphs = text.split('\n\n')
                if paragraphs and len(paragraphs[0]) > 20:
                    summary = paragraphs[0].strip()
                    if len(summary) > 250:
                        summary = summary[:247] + "..."
                else:
                    summary = text[:200] + "..." if len(text) > 200 else text
            
            # Find more detailed location info
            location = None
            if job_data.get('location'):
                location = job_data.get('location')
            elif 'address' in job_data and job_data['address'] and 'locality' in job_data['address']:
                if job_data['address'].get('region'):
                    location = f"{job_data['address'].get('locality')}, {job_data['address'].get('region')}"
                else:
                    location = job_data['address'].get('locality')
            
            # Extract company description
            company_description = None
            if 'description' in job_data:
                company_description = job_data.get('description')
            elif text:
                company_pattern = r'(?:about us|about the company|company overview)(?::|.{0,10})(.*?)(?:\n\n|\n\s*\n)'
                company_match = re.search(company_pattern, text, re.IGNORECASE | re.DOTALL)
                if company_match:
                    company_description = company_match.group(1).strip()
                    if len(company_description) > 150:
                        company_description = company_description[:147] + "..."
            
            # Determine availability (could be inferred from posting date)
            availability = "Available"
            posting_date = job_data.get('date')
            if not posting_date and 'estimatedDate' in job_data:
                posting_date = job_data['estimatedDate']
                
            if posting_date:
                try:
                    from dateutil import parser
                    posted_date = parser.parse(posting_date)
                    if (datetime.now(posted_date.tzinfo) - posted_date).days > 30:
                        availability = "May Be Filled"
                except:
                    pass
            
            # Extract application deadline
            application_deadline = None
            deadline_patterns = [
                r'(?:apply by|application deadline|closing date|deadline)(?::|is|.*?:)?\s*(\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
                r'(?:apply by|application deadline|closing date|deadline)(?::|is|.*?:)?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})(?:.*?deadline)'
            ]
            
            for pattern in deadline_patterns:
                deadline_match = re.search(pattern, text, re.IGNORECASE)
                if deadline_match:
                    application_deadline = deadline_match.group(1).strip()
                    break
            
            # Enhanced job type detection
            job_type = job_data.get('jobType')
            if not job_type and text:
                job_type_patterns = {
                    'full-time': r'full[- ]time|full time',
                    'part-time': r'part[- ]time|part time',
                    'contract': r'contract|contractor',
                    'freelance': r'freelance',
                    'internship': r'internship|intern',
                    'remote': r'remote|work from home|wfh',
                    'hybrid': r'hybrid'
                }
                
                for type_name, pattern in job_type_patterns.items():
                    if re.search(pattern, text.lower()):
                        job_type = type_name
                        break
            
            # Enhanced salary extraction
            salary_range = job_data.get('salaryRange')
            if not salary_range and text:
                # Look for salary patterns
                salary_patterns = [
                    r'\$\d{2,3}(?:,\d{3})+(?:\s*-\s*\$\d{2,3}(?:,\d{3})+)?(?:\s*per\s*year|\s*annually|\s*/\s*year)?',
                    r'\$\d{2,3}K\s*-\s*\$\d{2,3}K',
                    r'\d{2,3},\d{3}\s*-\s*\d{2,3},\d{3}\s*(?:USD|EUR|GBP)?'
                ]
                
                for pattern in salary_patterns:
                    salary_match = re.search(pattern, text)
                    if salary_match:
                        salary_range = salary_match.group(0)
                        break
            
            # Extract education requirements
            education_required = None
            edu_patterns = [
                r'(?:bachelor|master|phd|bs|ms|ba|degree|diploma)',
                r'(?:education|educational)(?:\s+requirements|\s+qualifications)?(?::|.{0,10})(.*?)(?:\n|\n\n)'
            ]
            for pattern in edu_patterns:
                edu_match = re.search(pattern, text, re.IGNORECASE)
                if edu_match:
                    # Extract the sentence containing the education requirement
                    sentence = re.findall(r'[^.!?]*(?:' + pattern + ')[^.!?]*[.!?]', text, re.IGNORECASE)
                    if sentence:
                        education_required = sentence[0].strip()
                        break
            
            # Extract experience requirements
            experience_required = None
            exp_patterns = [
                r'(\d+(?:-\d+)?\+?\s+years?(?:\s+of)?\s+experience)',
                r'experience(?::|.*?:)?\s+(\d+(?:-\d+)?\+?\s+years)'
            ]
            for pattern in exp_patterns:
                exp_match = re.search(pattern, text, re.IGNORECASE)
                if exp_match:
                    experience_required = exp_match.group(1).strip()
                    break
            
            # Extract job highlights
            job_highlights = []
            if text:
                # Look for bullet points after common headers
                highlight_pattern = r'(?:highlights|why you\'ll love this role|what you\'ll do|responsibilities|key responsibilities)(?::|.{0,10})(.*?)(?:\n\n\w|requirements|qualifications)'
                highlight_match = re.search(highlight_pattern, text, re.IGNORECASE | re.DOTALL)
                
                if highlight_match:
                    highlight_text = highlight_match.group(1).strip()
                    # Extract bullet points
                    if '•' in highlight_text:
                        points = [p.strip() for p in highlight_text.split('•') if p.strip()]
                    elif '-' in highlight_text:
                        points = [p.strip() for p in highlight_text.split('-') if p.strip()]
                    else:
                        # Split by new lines if no bullet points
                        points = [p.strip() for p in highlight_text.split('\n') if p.strip() and len(p.strip()) > 15]
                    
                    # Take just 3-5 key highlights
                    job_highlights = points[:5]
            
            # Look for images with better prioritization
            company_logo_url = None
            banner_image_url = None
            additional_images = []
            
            # First try to find a logo in the page data
            if 'logo' in job_data and job_data['logo'] and 'url' in job_data['logo']:
                company_logo_url = job_data['logo']['url']
            
            # Extract all images for further processing
            if 'images' in job_data:
                # Sort images by size, prioritizing larger ones for banner
                sorted_images = sorted(
                    [img for img in job_data.get('images', []) if img.get('url')],
                    key=lambda x: (x.get('width', 0) * x.get('height', 0)), 
                    reverse=True
                )
                
                for img in sorted_images:
                    img_url = img.get('url')
                    img_alt = img.get('alt', '').lower()
                    
                    # Skip tiny images and icons
                    if img.get('width', 0) < 100 or img.get('height', 0) < 100:
                        continue
                        
                    # Specific logo detection
                    if not company_logo_url and ('logo' in img_alt or 'company' in img_alt or 'brand' in img_alt):
                        company_logo_url = img_url
                        continue
                    
                    # Banner image - want a large, wide image
                    if not banner_image_url and img.get('width', 0) > 400 and img.get('width', 0) > img.get('height', 0):
                        banner_image_url = img_url
                        continue
                    
                    # Collect other useful images that might be relevant
                    additional_images.append(img_url)
            
            # If we have additional images but no banner, use the first additional image
            if not banner_image_url and additional_images:
                banner_image_url = additional_images[0]
                additional_images = additional_images[1:]
            
            # If no logo found but we have a company name, use a generated one
            if not company_logo_url and company:
                company_initial = company.strip()[0].upper() if company.strip() else "C"
                bg_color = "f8a5c2"  # Pink background
                company_logo_url = f"https://ui-avatars.com/api/?name={company_initial}&background={bg_color}&color=fff&size=128&bold=true&font-size=0.6"
            
            # Determine if job is women-friendly
            is_women_friendly_job = is_women_friendly(title, company, text)
            
            # Extract skills
            skills = extract_skills(text)
            
            # Extract qualifications
            qualifications = []
            qualifications_section = re.search(r'(?:requirements|qualifications|what you\'ll need)(?::|.*?:)(.*?)(?:responsibilities|benefits|about us|\n\n\w)', text.lower(), re.DOTALL | re.IGNORECASE)
            if qualifications_section:
                qual_text = qualifications_section.group(1).strip()
                # Extract bullet points
                if '•' in qual_text:
                    qualifications = [q.strip() for q in qual_text.split('•') if q.strip()]
                elif '-' in qual_text:
                    qualifications = [q.strip() for q in qual_text.split('-') if q.strip()]
                else:
                    # Split by new lines if no bullet points
                    qualifications = [q.strip() for q in qual_text.split('\n') if q.strip()]
                
                # Limit to top few qualifications
                qualifications = [q for q in qualifications[:5] if len(q) > 10]
            
            # Determine job category based on title and content
            category = "Tech Jobs"
            if any(keyword in title.lower() for keyword in ["data", "analyst", "scientist", "ml", "ai"]):
                category = "Data Science"
            elif any(keyword in title.lower() for keyword in ["developer", "engineer", "programmer", "code"]):
                category = "Software Engineering"
            elif any(keyword in title.lower() for keyword in ["design", "ux", "ui", "user experience"]):
                category = "Design"
            elif any(keyword in title.lower() for keyword in ["product", "manager", "owner"]):
                category = "Product"
            elif any(keyword in title.lower() for keyword in ["marketing", "growth", "seo"]):
                category = "Marketing"
            
            # Create job object with enhanced details
            job_info = JobBasic(
                title=title,
                company=company,
                location=location,
                job_type=job_type,
                posting_date=posting_date,
                salary_range=salary_range,
                application_url=url,
                is_women_friendly=is_women_friendly_job,
                skills=skills,
                company_logo_url=company_logo_url,
                banner_image_url=banner_image_url,
                summary=summary,
                company_description=company_description,
                availability=availability,
                category=category,
                education_required=education_required,
                experience_required=experience_required,
                application_deadline=application_deadline,
                job_highlights=job_highlights
            )
            
            # Cache the result
            job_cache[url] = job_info
            
            return job_info
        
        # Fallback if no detailed info is found
        logger.warning(f"No objects found in Diffbot response for {url}")
        return JobBasic(
            title="Job Listing",
            company="Unknown Company",
            application_url=url
        )
    
    except Exception as e:
        # Log the error but return a minimal job basic object
        logger.error(f"Error in fetch_job_info for {url}: {e}")
        return JobBasic(
            title="Job Listing",
            company="Unknown Company", 
            application_url=url
        )

def prefetch_job_details(urls: List[str]):
    """Prefetch and cache job details in the background"""
    for url in urls:
        try:
            if url not in job_cache:
                fetch_job_info(url)
        except Exception as e:
            logger.error(f"Error prefetching job details for {url}: {e}")

def fetch_job_details(url: str) -> JobDetail:
    """Fetch detailed job information from URL using Diffbot"""
    try:
        # First get basic info (which might already be cached)
        basic_info = fetch_job_info(url)
        
        API_URL = f"https://api.diffbot.com/v3/analyze?token={DIFFBOT_API_KEY}&url={url}&fields=links,meta,images,sentiment,facts&discussion=false&timeout=15000"
        
        response = requests.get(API_URL, timeout=15)
        
        # Handle unsuccessful responses
        if response.status_code != 200:
            # Return basic job info with empty additional fields
            return JobDetail(**basic_info.dict())
        
        data = response.json()
        
        # Extract additional details
        description = None
        qualifications = []
        skills_required = []
        benefits = []
        why_women_friendly = []
        additional_info = {}
        company_culture = None
        work_environment = None
        compensation_details = {}
        related_jobs = []
        
        if 'objects' in data and len(data['objects']) > 0:
            job_data = data['objects'][0]
            
            # Safely extract description
            description = job_data.get('text')
            
            # Try to extract qualifications and skills
            if description:
                # Extract sections based on common headers
                sections = re.split(r'\n\s*(?:Requirements|Qualifications|About the Role|Responsibilities|Benefits|What You\'ll Do|Who You Are|Company Culture|Work Environment)\s*\n', description, flags=re.IGNORECASE)
                
                if len(sections) > 1:
                    # First section is usually the job description
                    description = sections[0].strip()
                    
                    # Look for qualifications and requirements
                    for section in sections[1:]:
                        lines = [line.strip() for line in section.split('\n') if line.strip()]
                        
                        section_lower = section.lower()
                        if any(kw in section_lower for kw in ['qualif', 'require', 'who you are']):
                            qualifications.extend(lines[:5])  # Take up to 5 lines
                        
                        if any(kw in section_lower for kw in ['benefit', 'offer', 'perks']):
                            benefits.extend(lines[:5])  # Take up to 5 lines
                        
                        if any(kw in section_lower for kw in ['company culture', 'values', 'our culture']):
                            company_culture = '\n'.join(lines[:3])
                        
                        if any(kw in section_lower for kw in ['work environment', 'workplace', 'office']):
                            work_environment = '\n'.join(lines[:3])
                
                # Extract women-friendly aspects
                if basic_info.is_women_friendly:
                    for keyword in WOMEN_FRIENDLY_KEYWORDS:
                        if keyword in description.lower():
                            # Find the sentence containing the keyword
                            sentences = re.split(r'(?<=[.!?])\s+', description)
                            for sentence in sentences:
                                if keyword in sentence.lower():
                                    why_women_friendly.append(sentence.strip())
                                    break
                
                # Extract compensation details
                salary_pattern = r'salary(?:.*?)(\$[\d,]+(?:\s*-\s*\$[\d,]+)?(?:\s*(?:per|/)\s*(?:year|month|hour))?)'
                salary_match = re.search(salary_pattern, description, re.IGNORECASE)
                if salary_match:
                    compensation_details['salary'] = salary_match.group(1).strip()
                
                equity_pattern = r'equity(?:.*?)([\d\.]+%(?:\s*-\s*[\d\.]+%)?)'
                equity_match = re.search(equity_pattern, description, re.IGNORECASE)
                if equity_match:
                    compensation_details['equity'] = equity_match.group(1).strip()
                
                bonus_pattern = r'bonus(?:.*?)(\$[\d,]+|\d+%)'
                bonus_match = re.search(bonus_pattern, description, re.IGNORECASE)
                if bonus_match:
                    compensation_details['bonus'] = bonus_match.group(1).strip()
                
                # Try to extract related jobs
                if 'links' in job_data:
                    job_links = [link for link in job_data.get('links', []) if 'job' in link.get('url', '').lower()]
                    related_job_titles = []
                    
                    for link in job_links[:5]:  # Limit to 5 related jobs
                        title = link.get('title')
                        if title and len(title) < 100:  # Reasonable title length
                            related_job_titles.append(title)
                    
                    related_jobs = related_job_titles
            
            # Create detailed job info
            detailed_info = JobDetail(
                **basic_info.dict(),
                description=description,
                qualifications=qualifications or basic_info.qualifications,
                benefits=benefits,
                why_women_friendly=why_women_friendly,
                additional_info=additional_info,
                company_culture=company_culture,
                work_environment=work_environment,
                compensation_details=compensation_details,
                related_jobs=related_jobs
            )
            
            return detailed_info
        
        # Fallback to basic info
        return JobDetail(**basic_info.dict())
    
    except Exception as e:
        logger.error(f"Error in fetch_job_details for {url}: {e}")
        # Return basic job info with empty additional fields
        try:
            return JobDetail(**basic_info.dict())
        except:
            # In case basic_info is not defined due to an early exception
            return JobDetail(
                title="Job Listing",
                company="Unknown Company", 
                application_url=url
            )

@app.get("/api/job/{job_url:path}", response_model=JobDetail, tags=["Job Details"])
async def get_job_details(job_url: str):
    """Get detailed information about a specific job"""
    try:
        # URL decoding might be needed
        job_url = job_url.replace('___', '://')
        
        # Get detailed job info
        job_details = fetch_job_details(job_url)
        
        return job_details
    
    except Exception as e:
        logger.error(f"Error getting job details: {e}")
        raise HTTPException(status_code=404, detail=f"Job not found or error processing: {str(e)}")

@app.get("/api/statistics", tags=["Statistics"])
async def get_statistics():
    """Get statistics about the job search API"""
    return {
        "cache_size": len(job_cache),
        "api_version": "1.1.0",
        "women_friendly_companies_count": len(WOMEN_FRIENDLY_COMPANIES),
        "status": "healthy"
    }

@app.get("/api/job-searches", tags=["Job Search"])
async def get_job_searches(
    user_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    """Get job search history, optionally filtered by user ID"""
    try:
        # Build query
        query = {}
        if user_id:
            query["user_id"] = user_id
            
        # Fetch searches from MongoDB
        total = job_searches_collection.count_documents(query)
        searches = list(job_searches_collection.find(
            query, 
            sort=[("timestamp", DESCENDING)],
            skip=skip,
            limit=limit
        ))
        
        # Convert ObjectId to string
        for search in searches:
            search["_id"] = str(search["_id"])
            
        return {
            "searches": searches,
            "total": total,
            "limit": limit,
            "skip": skip
        }
    except Exception as e:
        logger.error(f"Error getting job searches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job-searches/{search_id}/results", tags=["Job Search"])
async def get_job_search_results(
    search_id: str,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    women_friendly_only: bool = Query(False)
):
    """Get job results for a specific search"""
    try:
        # Build query
        query = {"search_id": search_id}
        if women_friendly_only:
            query["is_women_friendly"] = True
            
        # Fetch jobs from MongoDB
        total = job_results_collection.count_documents(query)
        jobs = list(job_results_collection.find(
            query,
            sort=[("company", ASCENDING), ("title", ASCENDING)],
            skip=skip,
            limit=limit
        ))
        
        # Convert ObjectId to string
        for job in jobs:
            job["_id"] = str(job["_id"])
            
        return {
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "skip": skip
        }
    except Exception as e:
        logger.error(f"Error getting job search results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/saved-jobs", tags=["Job Search"])
async def get_saved_jobs(
    user_id: str = Query(..., description="User ID for retrieving saved jobs"),
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    """Get saved jobs for a specific user"""
    try:
        # This endpoint assumes you have a system for users saving jobs
        # For now, we'll return all women-friendly jobs from the user's searches
        
        # First get all searches by this user
        user_searches = job_searches_collection.find(
            {"user_id": user_id},
            projection={"search_id": 1}
        )
        
        search_ids = [search["search_id"] for search in user_searches]
        
        if not search_ids:
            return {
                "jobs": [],
                "total": 0,
                "limit": limit,
                "skip": skip
            }
        
        # Now get women-friendly jobs from those searches
        query = {
            "search_id": {"$in": search_ids},
            "is_women_friendly": True
        }
            
        # Fetch jobs from MongoDB
        total = job_results_collection.count_documents(query)
        jobs = list(job_results_collection.find(
            query,
            sort=[("stored_at", DESCENDING)],
            skip=skip,
            limit=limit
        ))
        
        # Convert ObjectId to string
        for job in jobs:
            job["_id"] = str(job["_id"])
            
        return {
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "skip": skip
        }
    except Exception as e:
        logger.error(f"Error getting saved jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/save", tags=["Job Search"])
async def save_job(
    user_id: str = Query(..., description="User ID saving the job"),
    job_data: JobBasic = Body(..., description="Job data to save")
):
    """Save a job for a user"""
    try:
        # Create a collection for saved jobs if it doesn't exist
        if "saved_jobs" not in db.list_collection_names():
            saved_jobs_collection = db["saved_jobs"]
            # Create indexes
            saved_jobs_collection.create_index([("user_id", ASCENDING)])
            saved_jobs_collection.create_index([("job_id", ASCENDING)])
            saved_jobs_collection.create_index([("application_url", ASCENDING)])
            saved_jobs_collection.create_index([("saved_at", DESCENDING)])
        else:
            saved_jobs_collection = db["saved_jobs"]
        
        # Generate a unique job ID if not provided
        job_id = str(uuid.uuid4())
        
        # Check if this job is already saved by this user
        existing_job = saved_jobs_collection.find_one({
            "user_id": user_id,
            "application_url": job_data.application_url
        })
        
        if existing_job:
            # Job already saved, remove it (toggle behavior)
            saved_jobs_collection.delete_one({"_id": existing_job["_id"]})
            return {
                "success": True,
                "message": "Job removed from saved jobs",
                "is_saved": False
            }
        
        # Prepare job data for saving
        job_dict = job_data.dict()
        job_dict["job_id"] = job_id
        job_dict["user_id"] = user_id
        job_dict["saved_at"] = datetime.utcnow()
        
        # Save to MongoDB
        saved_jobs_collection.insert_one(job_dict)
        
        return {
            "success": True,
            "message": "Job saved successfully",
            "job_id": job_id,
            "is_saved": True
        }
    except Exception as e:
        logger.error(f"Error saving job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/saved/{user_id}", tags=["Job Search"])
async def get_user_saved_jobs(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    """Get all saved jobs for a specific user"""
    try:
        # Get the saved_jobs collection
        saved_jobs_collection = db["saved_jobs"]
        
        # Query for this user's saved jobs
        total = saved_jobs_collection.count_documents({"user_id": user_id})
        saved_jobs = list(saved_jobs_collection.find(
            {"user_id": user_id},
            sort=[("saved_at", DESCENDING)],
            skip=skip,
            limit=limit
        ))
        
        # Convert ObjectId to string
        for job in saved_jobs:
            if "_id" in job:
                job["_id"] = str(job["_id"])
            # Convert datetime objects to ISO format strings
            if "saved_at" in job and isinstance(job["saved_at"], datetime):
                job["saved_at"] = job["saved_at"].isoformat()
        
        return {
            "success": True,
            "total": total,
            "limit": limit,
            "skip": skip,
            "jobs": saved_jobs
        }
    except Exception as e:
        logger.error(f"Error getting user saved jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/jobs/saved/{user_id}/{job_id}", tags=["Job Search"])
async def delete_saved_job(
    user_id: str,
    job_id: str
):
    """Delete a saved job for a user"""
    try:
        # Get the saved_jobs collection
        saved_jobs_collection = db["saved_jobs"]
        
        # Delete the saved job
        result = saved_jobs_collection.delete_one({
            "user_id": user_id,
            "job_id": job_id
        })
        
        if result.deleted_count == 0:
            return {
                "success": False,
                "message": "Job not found or already removed"
            }
        
        return {
            "success": True,
            "message": "Job removed from saved jobs"
        }
    except Exception as e:
        logger.error(f"Error deleting saved job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/analyze", tags=["Chat"])
async def analyze_chat_history(chat_history: List[ChatMessage]):
    """Analyze chat history to provide insights and recommendations"""
    try:
        # Use LLM to analyze the conversation
        llm = get_llm()
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_history]
        
        analysis_prompt = f"""
        Analyze the following conversation and provide:
        1. Key skills mentioned
        2. Career interests
        3. Potential job matches
        4. Recommended next steps
        5. Areas for improvement
        
        Conversation:
        {json.dumps(messages, indent=2)}
        """
        
        response = llm.invoke(analysis_prompt)
        return {"analysis": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/suggest", tags=["Chat"])
async def suggest_resources(chat_history: List[ChatMessage]):
    """Suggest learning resources based on chat history"""
    try:
        llm = get_llm()
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_history]
        
        suggestion_prompt = f"""
        Based on this conversation, suggest:
        1. Relevant online courses
        2. Books to read
        3. Communities to join
        4. Skills to develop
        5. Networking opportunities
        
        Conversation:
        {json.dumps(messages, indent=2)}
        """
        
        response = llm.invoke(suggestion_prompt)
        return {"suggestions": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/new", response_model=ChatResponse, tags=["Chat"])
async def chat_with_llama(
    request: ChatRequest,
    current_user_id: Optional[str] = Depends(get_current_user)
):
    try:
        # Validate request - but allow empty messages for just creating a session
        if request.message is None:
            raise HTTPException(
                status_code=400,
                detail="Message field cannot be missing"
            )

        # Use authenticated user ID if available, otherwise use the one from request
        # This ensures compatibility with both authenticated and unauthenticated requests
        actual_user_id = current_user_id or request.user_id

        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Check if session exists in MongoDB
        session_data = chat_sessions_collection.find_one({"session_id": session_id})
        
        if not session_data:
            # Create new session
            new_session = ChatSession(
                session_id=session_id,
                user_id=actual_user_id
            )
            chat_sessions_collection.insert_one(new_session.dict())
            logger.info(f"Created new chat session: {session_id}")
        else:
            # Check if the user has access to this session
            if session_data.get("user_id") and actual_user_id and session_data.get("user_id") != actual_user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have access to this chat session"
                )
            
            # Update session last active timestamp
            chat_sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "updated_at": datetime.utcnow(),
                    "is_active": True,
                    # Update user_id if it wasn't set before but we have it now
                    "user_id": actual_user_id or session_data.get("user_id")
                }}
            )
        
        # If message is empty, just return a session creation response
        if not request.message or not request.message.strip():
            return ChatResponse(
                session_id=session_id,
                response="Session created",
                message_id=str(uuid.uuid4()),
                jobs=[]
            )

        # Validate API key for chat with LLM
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is not configured")
            raise HTTPException(
                status_code=500,
                detail="Chat service is not properly configured. Please check GROQ_API_KEY."
            )

        try:
            # Initialize LLM
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama3-70b-8192",
                temperature=0.7,
                max_tokens=4000
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize chat service. Please try again."
            )

        # Get or create memory
        if session_id not in chat_memories:
            # Try to load from MongoDB
            chat_history = list(chat_messages_collection.find(
                {"session_id": session_id},
                sort=[("timestamp", 1)]
            ))
            
            memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                max_token_limit=2000
            )
            
            # Load previous messages to memory if they exist
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        memory.chat_memory.add_user_message(msg["content"])
                    else:
                        memory.chat_memory.add_ai_message(msg["content"])
            
            chat_memories[session_id] = memory
        else:
            memory = chat_memories[session_id]

        # Trim memory if needed (just for the LLM context window)
        if len(memory.chat_memory.messages) > 10:
            memory.chat_memory.messages = memory.chat_memory.messages[-10:]

        try:
            # Save user message to MongoDB
            user_message = ChatMessage(
                role="user",
                content=request.message,
                session_id=session_id,
                message_id=str(ObjectId())
            )
            chat_messages_collection.insert_one(user_message.dict())
            
            # Create conversation chain
            conversation = ConversationChain(
                llm=llm,
                memory=memory,
                prompt=PromptTemplate(
                    input_variables=["history", "input"],
                    template="""You are an AI assistant focused on supporting women in tech careers. Be helpful, encouraging, and professional.

Current conversation:
{history}
Human: {input}
Assistant:"""
                ),
                verbose=True
            )

            # Get response
            response = conversation.predict(input=request.message)
            
            if not response or not response.strip():
                raise ValueError("Empty response received from LLM")
                
            # Save AI response to MongoDB
            ai_message_id = str(ObjectId())
            ai_message = ChatMessage(
                role="assistant",
                content=response.strip(),
                session_id=session_id,
                message_id=ai_message_id
            )
            chat_messages_collection.insert_one(ai_message.dict())
            
            # Update session last message timestamp
            chat_sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {
                    "last_message_timestamp": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }}
            )

            return ChatResponse(
                response=response.strip(),
                session_id=session_id,
                message_id=ai_message_id
            )

        except ValueError as ve:
            logger.error(f"Value error in conversation: {str(ve)}")
            raise HTTPException(
                status_code=500,
                detail="Received an invalid response. Please try again."
            )
        except Exception as e:
            logger.error(f"Error in conversation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process your message. Please try again."
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )

@app.get("/api/chat/history/{session_id}", response_model=List[ChatMessage], tags=["Chat"])
async def get_chat_history(
    session_id: str, 
    limit: int = 50, 
    skip: int = 0,
    current_user_id: Optional[str] = Depends(get_current_user)
):
    """Get chat history for a specific session"""
    try:
        # Validate session_id format
        if not session_id or len(session_id) < 8:
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID format"
            )
            
        # Check if session exists
        session = chat_sessions_collection.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found"
            )

        # Check if the user has access to this session
        if session.get("user_id") and current_user_id and session.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this chat session"
            )
            
        # Get messages from MongoDB
        messages = list(chat_messages_collection.find(
            {"session_id": session_id},
            sort=[("timestamp", -1)],  # Sort by timestamp descending (newest first)
            skip=skip,
            limit=limit
        ))
        
        # Convert to ChatMessage models and return
        chat_messages = []
        for msg in messages:
            # Convert ObjectId to str
            if "_id" in msg:
                msg["_id"] = str(msg["_id"])
                
            chat_messages.append(ChatMessage(**msg))
            
        return sorted(chat_messages, key=lambda x: x.timestamp)  # Return sorted by timestamp ascending
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chat history"
        )

@app.get("/api/chat/sessions", response_model=List[ChatSession], tags=["Chat"])
async def get_chat_sessions(
    limit: int = 20, 
    skip: int = 0, 
    active_only: bool = True,
    current_user_id: Optional[str] = Depends(get_current_user),
    user_id: Optional[str] = None
):
    """Get list of chat sessions for a user"""
    try:
        # Prefer authenticated user_id over query parameter
        actual_user_id = current_user_id or user_id
        
        # Build query
        query = {}
        if actual_user_id:
            query["user_id"] = actual_user_id
        if active_only:
            query["is_active"] = True
            
        # Get sessions from MongoDB
        sessions = list(chat_sessions_collection.find(
            query,
            sort=[("updated_at", -1)],  # Sort by last updated time
            skip=skip,
            limit=limit
        ))
        
        # Convert to ChatSession models
        chat_sessions = []
        for session in sessions:
            # Convert ObjectId to str
            if "_id" in session:
                session["_id"] = str(session["_id"])
                
            # Count messages for this session
            message_count = chat_messages_collection.count_documents({"session_id": session["session_id"]})
            
            # Create session object with additional info
            session_obj = ChatSession(**session)
            session_obj.message_count = message_count
            
            chat_sessions.append(session_obj)
            
        return chat_sessions
        
    except Exception as e:
        logger.error(f"Error getting chat sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chat sessions"
        )

@app.delete("/api/chat/sessions/{session_id}", tags=["Chat"])
async def delete_session(
    session_id: str, 
    delete_messages: bool = False,
    current_user_id: Optional[str] = Depends(get_current_user)
):
    """Delete a chat session"""
    try:
        # Check if session exists and belongs to the user
        session = chat_sessions_collection.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
            
        # Check if the user has access to this session
        if session.get("user_id") and current_user_id and session.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this chat session"
            )
        
        # Delete session from MongoDB
        result = chat_sessions_collection.delete_one({"session_id": session_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
            
        # Delete from in-memory storage as well
        if session_id in chat_memories:
            del chat_memories[session_id]
            
        # Optionally delete associated messages
        if delete_messages:
            chat_messages_collection.delete_many({"session_id": session_id})
            
        return {"status": "success", "message": "Session deleted"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete session"
        )

@app.delete("/api/chat/memory/{session_id}", tags=["Chat"])
async def clear_memory(
    session_id: str,
    current_user_id: Optional[str] = Depends(get_current_user)
):
    """Clear memory for a chat session"""
    try:
        # Check if session exists and belongs to the user
        session = chat_sessions_collection.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
            
        # Check if the user has access to this session
        if session.get("user_id") and current_user_id and session.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this chat session"
            )
            
        if session_id in chat_memories:
            del chat_memories[session_id]
            
        # Also update session in MongoDB
        chat_sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False}}
        )
        
        return {"status": "success", "message": "Memory cleared"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear memory"
        )

@app.get("/api/auth/verify", tags=["Auth"])
async def verify_auth(current_user_id: Optional[str] = Depends(get_current_user)):
    """Verify authentication and return user ID if authenticated"""
    if not current_user_id:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )
    
    return {
        "authenticated": True,
        "user_id": current_user_id
    }

# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
