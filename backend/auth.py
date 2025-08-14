# backend/auth.py
"""
Lightweight auth for KelpGPT (Windows/SQLite-ready)
- 6-digit passcodes (bcrypt-hashed in DB)
- Server-side sessions (sha256-hashed tokens)
- Admin utilities for creating users and managing codes/sessions
"""

from __future__ import annotations
import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

import bcrypt
from sqlalchemy.orm import Session as SASession

from .db import SessionLocal
from .models import User, LoginCode, Session as DBSession


# ==========
# Internals
# ==========

PASSCODE_RE = re.compile(r"^\d{6}$")  # exactly 6 digits

def _now_utc() -> datetime:
    return datetime.utcnow()

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ======================
# Login code management
# ======================

def hash_login_code(code_plain: str) -> str:
    """Hash a login code using bcrypt."""
    return bcrypt.hashpw(code_plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def generate_passcode() -> str:
    """Generate a zero-padded 6-digit numeric passcode."""
    return f"{secrets.randbelow(1_000_000):06d}"

def set_login_code(user_id: int, code_plain: str, *, expires_in_days: int = 90) -> None:
    """
    Store a login code (any string) for a user. Prefer set_login_passcode() for 6-digit codes.
    """
    db: SASession = SessionLocal()
    try:
        u = db.get(User, user_id)
        if not u:
            raise ValueError(f"User {user_id} not found")
        lc = LoginCode(
            user_id=user_id,
            code_hash=hash_login_code(code_plain),
            expires_at=_now_utc() + timedelta(days=expires_in_days),
            is_revoked=False,
        )
        db.add(lc)
        db.commit()
    finally:
        db.close()

def set_login_passcode(user_id: int, passcode6: str, *, expires_in_days: int = 90) -> None:
    """Store a bcrypt-hashed 6-digit numeric passcode for the user."""
    if not PASSCODE_RE.match(passcode6):
        raise ValueError("Passcode must be exactly 6 digits (000000–999999).")
    set_login_code(user_id, passcode6, expires_in_days=expires_in_days)

def revoke_login_codes(user_id: int) -> int:
    """Revoke all existing login codes for a user. Returns number revoked."""
    db: SASession = SessionLocal()
    try:
        q = db.query(LoginCode).filter(LoginCode.user_id == user_id, LoginCode.is_revoked == False)  # noqa: E712
        n = 0
        for lc in q.all():
            lc.is_revoked = True
            n += 1
        db.commit()
        return n
    finally:
        db.close()

def verify_login_code(user_id: int, code_plain: str) -> bool:
    """Check the most recent, non-revoked login code (by id desc)."""
    db: SASession = SessionLocal()
    try:
        lc = (
            db.query(LoginCode)
            .filter(LoginCode.user_id == user_id, LoginCode.is_revoked == False)  # noqa: E712
            .order_by(LoginCode.id.desc())
            .first()
        )
        if not lc:
            return False
        if lc.expires_at and lc.expires_at < _now_utc():
            return False
        return bcrypt.checkpw(code_plain.encode("utf-8"), lc.code_hash.encode("utf-8"))
    finally:
        db.close()


# ===========
# Users CRUD
# ===========

def create_user(username: str, *, email: Optional[str] = None, role: str = "user") -> User:
    """Create a new user (active). Does NOT set a login code."""
    db: SASession = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise ValueError(f"Username '{username}' already exists")
        u = User(username=username.strip(), email=(email or "").strip() or None, role=role, is_active=True)
        db.add(u)
        db.commit()
        db.refresh(u)
        return u
    finally:
        db.close()

def get_user_by_username(username: str) -> Optional[User]:
    db: SASession = SessionLocal()
    try:
        return db.query(User).filter(User.username == username, User.is_active == True).first()  # noqa: E712
    finally:
        db.close()


# ==================
# Session management
# ==================

def create_session(user_id: int, *, ttl_hours: int = 24) -> str:
    """
    Create a server-side session. Returns raw token; DB stores SHA-256 hash only.
    """
    token = secrets.token_urlsafe(32)  # opaque, random
    token_hash = _sha256(token)
    db: SASession = SessionLocal()
    try:
        sess = DBSession(
            user_id=user_id,
            token_hash=token_hash,
            created_at=_now_utc(),
            expires_at=_now_utc() + timedelta(hours=ttl_hours),
        )
        db.add(sess)
        db.commit()
        return token
    finally:
        db.close()

def validate_session(token: str) -> Optional[int]:
    """
    Return user_id if token is valid & unexpired; else None.
    """
    if not token:
        return None
    th = _sha256(token)
    db: SASession = SessionLocal()
    try:
        s = db.query(DBSession).filter(DBSession.token_hash == th).first()
        if not s or (s.expires_at and s.expires_at < _now_utc()):
            return None
        return s.user_id
    finally:
        db.close()

def rotate_session(token: str, *, ttl_hours: int = 24) -> Optional[str]:
    """
    Invalidate an existing session and issue a new token. Returns new token or None.
    """
    if not token:
        return None
    old_hash = _sha256(token)
    db: SASession = SessionLocal()
    try:
        s = db.query(DBSession).filter(DBSession.token_hash == old_hash).first()
        if not s:
            return None
        uid = s.user_id
        db.delete(s)
        db.commit()
        return create_session(uid, ttl_hours=ttl_hours)
    finally:
        db.close()

def end_session(token: str) -> bool:
    """Invalidate a single session token."""
    if not token:
        return False
    th = _sha256(token)
    db: SASession = SessionLocal()
    try:
        s = db.query(DBSession).filter(DBSession.token_hash == th).first()
        if not s:
            return False
        db.delete(s)
        db.commit()
        return True
    finally:
        db.close()

def end_all_sessions(user_id: int) -> int:
    """Invalidate all sessions for a user. Returns number deleted."""
    db: SASession = SessionLocal()
    try:
        q = db.query(DBSession).filter(DBSession.user_id == user_id)
        n = q.count()
        q.delete(synchronize_session=False)
        db.commit()
        return n
    finally:
        db.close()


# ==============================
# Convenience (UI integration)
# ==============================

def login_with_code(username: str, code_plain: str, *, ttl_hours: int = 24) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Verify a login code and return (ok, token, user_id).
    Use set_login_passcode() to enforce 6-digit codes for users.
    """
    u = get_user_by_username(username)
    if not u:
        return (False, None, None)
    if not verify_login_code(u.id, code_plain):
        return (False, None, None)
    token = create_session(u.id, ttl_hours=ttl_hours)
    return (True, token, u.id)

def current_user_id_from_token(token: Optional[str]) -> Optional[int]:
    """Thin wrapper for validate_session(), friendly in Streamlit."""
    if not token:
        return None
    return validate_session(token)


# ==========================
# Optional admin conveniences
# ==========================

def upsert_user_with_code(username: str, code_plain: str, *, email: Optional[str] = None, role: str = "user", expires_in_days: int = 90) -> User:
    """
    Create the user if missing, then set (add) a login code (any string).
    Prefer passing a 6-digit code if you're standardizing on passcodes.
    """
    db: SASession = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username).first()
        if not u:
            u = create_user(username, email=email, role=role)
        set_login_code(u.id, code_plain, expires_in_days=expires_in_days)
        return u
    finally:
        db.close()


__all__ = [
    # user & codes
    "create_user", "get_user_by_username",
    "set_login_code", "set_login_passcode", "revoke_login_codes",
    "hash_login_code", "generate_passcode", "verify_login_code",
    # sessions
    "create_session", "validate_session", "rotate_session",
    "end_session", "end_all_sessions",
    # ui helpers
    "login_with_code", "current_user_id_from_token",
    # admin
    "upsert_user_with_code",
]

