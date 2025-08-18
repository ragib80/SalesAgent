from typing import Optional, Literal, Dict
from pydantic import BaseModel, Field

class DateRange(BaseModel):
    start: Optional[str] = Field(None, description="YYYY-MM-DD")
    end: Optional[str] = Field(None, description="YYYY-MM-DD")
    granularity: Literal["day","week","month","quarter","year"] = "day"

class Scope(BaseModel):
    region: Optional[str] = None
    zone: Optional[str] = None
    depo: Optional[str] = None         # gsber (code or human name accepted)
    dealer: Optional[str] = None

class Entities(BaseModel):
    brand: Optional[str] = None
    product: Optional[str] = None

class Frame(BaseModel):
    date: DateRange = DateRange()
    scope: Scope = Scope()
    entities: Entities = Entities()
    metric: Literal["Revenue","Volume","InvoiceCount","AvgSellingPrice"] = "Revenue"
    group_by: Optional[Literal["dealer","brand","product","depo","zone","region","date"]] = None
    limit: int = 10
    order: Literal["asc","desc"] = "desc"
    compare: Optional[Dict] = None
    units: Literal["BDT","units"] = "BDT"
    filters: Dict[str,str] = {}
    user_scope_ok: bool = True

class FrameDelta(BaseModel):
    # Only include keys that changed; omissions mean "reuse previous"
    date: Optional[DateRange] = None
    scope: Optional[Scope] = None
    entities: Optional[Entities] = None
    metric: Optional[str] = None
    group_by: Optional[str] = None
    limit: Optional[int] = None
    order: Optional[Literal["asc","desc"]] = None
    compare: Optional[Dict] = None
    filters: Optional[Dict[str,str]] = None
