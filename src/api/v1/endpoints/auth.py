from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
import sqlite3
from pathlib import Path
import os
import hashlib
import uuid
import datetime
from typing import Optional


DB_PATH = Path("local_storage") / "users.db"


def ensure_db():
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	cur.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE NOT NULL,
			password_hash TEXT NOT NULL,
			token TEXT,
			created_at TEXT NOT NULL
		)
		"""
	)
	conn.commit()
	conn.close()


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
	# returns salt_hex$hash_hex
	if salt is None:
		salt = os.urandom(16)
	dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
	return salt.hex() + "$" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
	try:
		salt_hex, hash_hex = stored.split("$")
		salt = bytes.fromhex(salt_hex)
		dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
		return dk.hex() == hash_hex
	except Exception:
		return False


def create_user(username: str, password: str):
	ensure_db()
	ph = hash_password(password)
	created_at = datetime.datetime.utcnow().isoformat()
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	try:
		cur.execute(
			"INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
			(username, ph, created_at),
		)
		conn.commit()
		user_id = cur.lastrowid
	except sqlite3.IntegrityError:
		conn.close()
		raise HTTPException(status_code=400, detail="username already exists")
	conn.close()
	return user_id


def set_token_for_user(user_id: int, token: str):
	ensure_db()
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	cur.execute("UPDATE users SET token = ? WHERE id = ?", (token, user_id))
	conn.commit()
	conn.close()


def authenticate_user(username: str, password: str):
	ensure_db()
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	cur.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
	row = cur.fetchone()
	conn.close()
	if not row:
		return None
	user_id, password_hash = row
	if verify_password(password, password_hash):
		# create token
		token = uuid.uuid4().hex
		set_token_for_user(user_id, token)
		return {"id": user_id, "token": token}
	return None


def get_user_by_token(token: str):
	if not token:
		return None
	ensure_db()
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()
	cur.execute("SELECT id, username FROM users WHERE token = ?", (token,))
	row = cur.fetchone()
	conn.close()
	if not row:
		return None
	return {"id": row[0], "username": row[1]}


class RegisterRequest(BaseModel):
	username: str
	password: str


class AuthResponse(BaseModel):
	id: int
	token: str


router = APIRouter(prefix="/auth", tags=["auth"]) 

@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
	"""Register a new user. Returns id and token."""
	user_id = create_user(req.username, req.password)
	token = uuid.uuid4().hex
	set_token_for_user(user_id, token)
	return {"id": user_id, "token": token}


@router.post("/login", response_model=AuthResponse)
def login(req: RegisterRequest):
	"""Login existing user and return id and token."""
	auth = authenticate_user(req.username, req.password)
	if not auth:
		raise HTTPException(status_code=401, detail="invalid credentials")
	return {"id": auth["id"], "token": auth["token"]}


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
	if not authorization:
		return None
	parts = authorization.split()
	if len(parts) == 2 and parts[0].lower() == "bearer":
		return parts[1]
	return None


def get_current_user(token: Optional[str] = Header(None, alias="Authorization")):
	"""FastAPI dependency: pass Authorization: Bearer <token> header. Returns user dict or raises 401."""
	raw = token
	bearer = _extract_bearer(raw)
	user = get_user_by_token(bearer)
	if not user:
		raise HTTPException(status_code=401, detail="invalid or missing token")
	return user


def get_current_user_id(authorization: Optional[str] = Header(None, alias="Authorization")) -> int:
	"""Dependency that returns just the user id for downstream processing."""
	user = get_current_user(authorization)
	return user["id"]


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
	return {"id": user["id"], "username": user["username"]}


# Export the dependency names so other modules can `from .auth import get_current_user_id`
__all__ = ["router", "get_current_user_id", "get_current_user"]
