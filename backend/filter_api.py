from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:admin123@my-postgres-postgresql.database.svc.cluster.local:5432/postgres",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class FilterRuleModel(Base):
    __tablename__ = "filter_rules"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    pattern = Column(String, nullable=False)

Base.metadata.create_all(engine)

app = FastAPI()

class FilterRule(BaseModel):
    id: int | None = None
    name: str
    pattern: str

@app.get("/filters", response_model=List[FilterRule])
def get_filters():
    with SessionLocal() as session:
        rules = session.query(FilterRuleModel).all()
        return [FilterRule(id=r.id, name=r.name, pattern=r.pattern) for r in rules]

@app.put("/filters")
def replace_filters(rules: List[FilterRule]):
    with SessionLocal() as session:
        session.query(FilterRuleModel).delete()
        session.bulk_save_objects([FilterRuleModel(name=r.name, pattern=r.pattern) for r in rules])
        session.commit()
    return {"status": "ok"}

@app.post("/filters", response_model=FilterRule)
def add_filter(rule: FilterRule):
    with SessionLocal() as session:
        obj = FilterRuleModel(name=rule.name, pattern=rule.pattern)
        session.add(obj)
        session.commit()
    return rule

@app.delete("/filters/{rule_id}")
def delete_filter(rule_id: int):
    with SessionLocal() as session:
        deleted = session.query(FilterRuleModel).filter(FilterRuleModel.id == rule_id).delete()
        session.commit()
        if not deleted:
            raise HTTPException(status_code=404, detail="Filter not found")
    return {"status": "deleted"}
