"""
Reports API Endpoints.
List, view, and export multi-agent analysis reports to PDF or JSON.
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_db, get_current_user
from app.db.models import User, Report, Query, Dataset
from app.schemas.query import ReportResponse
from app.services.report_service import generate_pdf_report

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.get("", response_model=list[dict])
async def list_reports(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all generated reports for the current user across all datasets.
    """
    result = await db.execute(
        select(Report)
        .join(Query, Report.query_id == Query.id)
        .where(Query.user_id == current_user.id)
        .options(selectinload(Report.query).selectinload(Query.dataset))
        .order_by(Report.created_at.desc())
    )
    reports = result.scalars().all()

    items = []
    for r in reports:
        items.append({
            "id": r.id,
            "query_id": r.query_id,
            "question": r.query.question if r.query else "",
            "dataset_name": r.query.dataset.name if (r.query and r.query.dataset) else "",
            "dataset_id": r.query.dataset_id if r.query else None,
            "answer": r.answer,
            "insights": r.insights,
            "agents_used": r.agents_used,
            "created_at": r.created_at,
        })
    return items


@router.get("/{report_id}/export/pdf")
async def export_report_pdf(
    report_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Download a report in PDF format.
    """
    result = await db.execute(
        select(Report)
        .join(Query, Report.query_id == Query.id)
        .where(Report.id == report_id, Query.user_id == current_user.id)
        .options(selectinload(Report.query).selectinload(Query.dataset))
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    question = report.query.question if report.query else "Data Query"
    ds_name = report.query.dataset.name if (report.query and report.query.dataset) else "Dataset"

    report_dict = {
        "answer": report.answer,
        "insights": report.insights,
        "agents_used": report.agents_used,
        "data_used": report.data_used,
    }

    pdf_buffer = generate_pdf_report(report_dict, question, ds_name)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id.hex[:8]}.pdf"'},
    )


@router.get("/{report_id}/export/json")
async def export_report_json(
    report_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Export report in JSON format.
    """
    result = await db.execute(
        select(Report)
        .join(Query, Report.query_id == Query.id)
        .where(Report.id == report_id, Query.user_id == current_user.id)
        .options(selectinload(Report.query).selectinload(Query.dataset))
    )
    report = result.scalar_one_or_none()

    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    payload = {
        "report_id": str(report.id),
        "question": report.query.question if report.query else "",
        "dataset_name": report.query.dataset.name if (report.query and report.query.dataset) else "",
        "answer": report.answer,
        "insights": report.insights,
        "agents_used": report.agents_used,
        "data_used": report.data_used,
        "created_at": report.created_at.isoformat(),
    }

    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id.hex[:8]}.json"'},
    )
