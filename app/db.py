import os
import motor.motor_asyncio
from datetime import datetime
from bson.objectid import ObjectId
from typing import List

_db_client = None
_db = None
_hist_collection = None

async def init_db(mongo_uri: str, db_name: str):
    global _db_client, _db, _hist_collection
    _db_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    _db = _db_client[db_name]
    _hist_collection = _db["history"]
    # create index on timestamp for faster retrieval
    await _hist_collection.create_index("timestamp", expireAfterSeconds=False)

async def insert_history(record: dict):
    """
    Insert a record into history collection. Adds timestamp and returns inserted id.
    """
    rec = record.copy()
    rec["timestamp"] = datetime.utcnow().isoformat()
    res = await _hist_collection.insert_one(rec)
    return str(res.inserted_id)

async def get_history(limit: int = 50, skip: int = 0):
    """
    Retrieve history documents ordered by most recent first.
    """
    cursor = _hist_collection.find({}).sort("timestamp", -1).skip(int(skip)).limit(int(limit))
    items = []
    async for doc in cursor:
        items.append({
            "id": str(doc.get("_id")),
            "url": doc.get("url"),
            "label_index": int(doc.get("label_index", 0)),
            "label": doc.get("label"),
            "confidence": float(doc.get("confidence", 0.0)),
            "explanation": doc.get("explanation"),
            "timestamp": doc.get("timestamp")
        })
    return items
