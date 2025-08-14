from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from datetime import datetime
# ...existing imports and Base...

class RegistrationRequest(Base):
    __tablename__ = "registration_requests"
    id = Column(Integer, primary_key=True)
    username = Column(String(120), nullable=False)          # requested username
    email = Column(String(255))                              # optional
    notes = Column(String(2000))                             # why they need access, team, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    approved = Column(Boolean, default=False)
    processed_by = Column(String(120))                       # your admin username
    meta_json = Column(JSON, default=dict)                   # freeform (e.g., org, role)
    
