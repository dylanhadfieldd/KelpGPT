# backend/models.py
"""
SQLAlchemy models for KelpGPT:
- Users & 6-digit login codes (bcrypt-hashed)
- Server-side sessions (sha256-hashed tokens)
- User preferences (tone, citations, temperature)
- Prompt history (for analytics & personalization)
- Registration requests (ticket flow -> admin approval)
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON, Float, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ===============
# Core user model
# ===============

class User(Base):
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True)
    username: str = Column(String(120), unique=True, nullable=False, index=True)
    email: Optional[str] = Column(String(255), unique=True)
    role: str = Column(String(50), default="user")
    avatar_path: Optional[str] = Column(String(512))
    is_active: bool = Column(Boolean, default=True)

    # relationships
    prefs = relationship("UserPrefs", uselist=False, back_populates="user", cascade="all, delete-orphan")
    login_codes = relationship("LoginCode", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    history = relationship("PromptHistory", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} active={self.is_active}>"


# =====================
# Login codes (bcrypt)
# =====================

class LoginCode(Base):
    __tablename__ = "login_codes"

    id: int = Column(Integer, primary_key=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    code_hash: str = Column(String(200), nullable=False)             # bcrypt hash
    expires_at: Optional[datetime] = Column(DateTime)
    is_revoked: bool = Column(Boolean, default=False)

    user = relationship("User", back_populates="login_codes")

    __table_args__ = (
        Index("ix_login_codes_user_valid", "user_id", "is_revoked"),
    )


# ===========================
# Sessions (server-side hash)
# ===========================

class Session(Base):
    __tablename__ = "sessions"

    id: int = Column(Integer, primary_key=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash: str = Column(String(200), nullable=False, unique=True)  # sha256 of token
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    expires_at: Optional[datetime] = Column(DateTime, index=True)

    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index("ix_sessions_token_hash", "token_hash"),
    )


# ==========================
# User preferences (1:1)
# ==========================

class UserPrefs(Base):
    __tablename__ = "user_prefs"

    id: int = Column(Integer, primary_key=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), unique=True, index=True)
    tone: str = Column(String(40), default="technical")          # 'technical','concise','friendly','detailed'
    cite_style: str = Column(String(20), default="inline")       # 'inline','APA','none'
    default_temp: float = Column(Float, default=0.2)
    json_blob = Column(JSON, default=dict)                       # counters/flags: e.g. asked_for_citations_n

    user = relationship("User", back_populates="prefs")


# ========================
# Prompt history (many:1)
# ========================

class PromptHistory(Base):
    __tablename__ = "prompt_history"

    id: int = Column(Integer, primary_key=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), index=True)
    ts: datetime = Column(DateTime, default=datetime.utcnow, index=True)
    input_text: str = Column(Text)
    response_style = Column(JSON)       # snapshot of applied prefs (tone/temp/etc)
    tags = Column(JSON)                 # e.g., ['RAG','aquaculture']

    user = relationship("User", back_populates="history")


# ============================
# Registration request tickets
# ============================

class RegistrationRequest(Base):
    __tablename__ = "registration_requests"

    id: int = Column(Integer, primary_key=True)
    username: str = Column(String(120), nullable=False)     # requested username
    email: Optional[str] = Column(String(255))              # optional
    notes: Optional[str] = Column(String(2000))             # reason/team/etc
    created_at: datetime = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at: Optional[datetime] = Column(DateTime)
    approved: bool = Column(Boolean, default=False)
    processed_by: Optional[str] = Column(String(120))       # admin username
    meta_json = Column(JSON, default=dict)                  # freeform metadata

    def __repr__(self) -> str:
        status = "pending" if not self.processed_at else ("approved" if self.approved else "denied")
        return f"<RegistrationRequest id={self.id} user={self.username!r} status={status}>"
