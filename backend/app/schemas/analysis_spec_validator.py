"""
AnalysisSpec Validator.
Authoritative semantic validator for AnalysisSpec (QueryContract) against the active dataset schema.
Enforces:
1. Schema Field Existence (dimensions, measures, group_by, series_dimension, filters)
2. Aggregation Validity against Datatypes (COUNT(*), SUM/AVG numeric only, MIN/MAX real columns)
3. Filter Validity (columns exist, categorical value alignment)
4. Grouping & Series Preservation (prevent scalar collapse when grouped)
5. Chart Type Compatibility (e.g. Histogram requires numeric, Bar/Pie for categorical)
"""

from typing import Any, Optional
import re
import logging

from app.schemas.query_contract import QueryContract, FilterSpec
from app.utils.schema_resolver import SchemaResolver, is_temporal_column

logger = logging.getLogger(__name__)


class AnalysisSpecValidator:
    """
    Deterministic backend validator for the canonical AnalysisSpec (QueryContract).
    Enforces contract correctness before SQL generation and visualization planning.
    """

    ALLOWED_AGGREGATIONS = {"COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"}

    @classmethod
    def validate(
        cls,
        contract: QueryContract,
        schema: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate AnalysisSpec against active dataset schema without modifying it.
        Returns: (is_valid: bool, errors: list[str])
        """
        errors: list[str] = []

        if not contract.is_related_to_dataset or contract.intent == "UNRELATED":
            return True, []

        raw_cols = schema.get("columns", []) if isinstance(schema, dict) else []
        schema_col_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in raw_cols if c]
        cols_lower = {c.lower(): c for c in schema_col_names}

        # 1. FIELD VALIDATION
        # 1a. Dimensions & Group By
        for dim in contract.dimensions:
            if dim and dim.lower() not in cols_lower:
                resolved = SchemaResolver.resolve_column(dim, schema_col_names, schema)
                if not resolved:
                    errors.append(f"Dimension '{dim}' does not exist in active dataset schema.")

        for grp in contract.group_by:
            if grp and grp.lower() not in cols_lower:
                resolved = SchemaResolver.resolve_column(grp, schema_col_names, schema)
                if not resolved:
                    errors.append(f"Grouping field '{grp}' does not exist in active dataset schema.")

        # 1b. Series Dimension
        if contract.series_dimension:
            if contract.series_dimension.lower() not in cols_lower:
                resolved = SchemaResolver.resolve_column(contract.series_dimension, schema_col_names, schema)
                if not resolved:
                    errors.append(f"Series dimension '{contract.series_dimension}' does not exist in active dataset schema.")

        # 1c. Measure Column
        if contract.measure_column:
            mc = contract.measure_column.strip()
            if mc not in ("*", "total_records"):
                sub_ids = re.findall(r'[a-zA-Z_]\w*', mc)
                for sid in sub_ids:
                    if sid.lower() not in cols_lower and sid != "total_records":
                        resolved = SchemaResolver.resolve_column(sid, schema_col_names, schema)
                        if not resolved:
                            errors.append(f"Measure column '{sid}' does not exist in active dataset schema.")

        # 1d. Filters
        for f in contract.filters:
            f_col = getattr(f, "column", None)
            if f_col and f_col.lower() not in cols_lower:
                resolved = SchemaResolver.resolve_column(f_col, schema_col_names, schema)
                if not resolved:
                    errors.append(f"Filter column '{f_col}' does not exist in active dataset schema.")

        # 1e. Calculation Operands
        if contract.calculation:
            for op in contract.calculation.operands:
                op_meas = getattr(op, "measure_column", None)
                if op_meas and op_meas not in ("*", "total_records") and op_meas.lower() not in cols_lower:
                    resolved = SchemaResolver.resolve_column(op_meas, schema_col_names, schema)
                    if not resolved:
                        errors.append(f"Calculation operand measure '{op_meas}' does not exist in active dataset schema.")
                for op_f in getattr(op, "filters", []) or []:
                    op_f_col = getattr(op_f, "column", None)
                    if op_f_col and op_f_col.lower() not in cols_lower:
                        resolved = SchemaResolver.resolve_column(op_f_col, schema_col_names, schema)
                        if not resolved:
                            errors.append(f"Calculation operand filter column '{op_f_col}' does not exist in active dataset schema.")

        # 2. AGGREGATION VALIDATION
        if contract.aggregation:
            agg_upper = contract.aggregation.upper()
            if agg_upper not in cls.ALLOWED_AGGREGATIONS:
                errors.append(f"Invalid aggregation '{contract.aggregation}'. Allowed: {', '.join(sorted(cls.ALLOWED_AGGREGATIONS))}.")

            if agg_upper in ("SUM", "AVG"):
                if not contract.measure_column or contract.measure_column in ("*", "total_records"):
                    errors.append(f"Aggregation {agg_upper} requires an actual numeric column, but got '{contract.measure_column}'.")
                else:
                    is_num = SchemaResolver.is_numeric_column(contract.measure_column, schema)
                    if not is_num:
                        errors.append(f"Aggregation {agg_upper} requires a numeric column, but '{contract.measure_column}' is non-numeric.")

            if agg_upper in ("MIN", "MAX"):
                if not contract.measure_column or contract.measure_column in ("*", "total_records"):
                    errors.append(f"Aggregation {agg_upper} requires an actual column, but got '{contract.measure_column}'.")

        # 3. GROUPING & SERIES INTEGRITY
        if contract.group_by:
            if contract.answer_type == "SCALAR":
                errors.append("AnalysisSpec has non-empty group_by but answer_type is SCALAR.")

        if contract.series_dimension:
            if contract.series_dimension not in contract.group_by:
                errors.append(f"Series dimension '{contract.series_dimension}' must be included in group_by {contract.group_by}.")

        # 4. CHART COMPATIBILITY
        if contract.suggested_chart_type:
            chart = contract.suggested_chart_type.lower()
            if chart == "histogram":
                target = contract.x_axis or contract.measure_column or (contract.dimensions[0] if contract.dimensions else None)
                if target:
                    is_num = SchemaResolver.is_numeric_column(target, schema)
                    if not is_num:
                        errors.append(f"Histogram requires a numeric continuous dimension, but '{target}' is categorical.")

            elif chart == "multi_line":
                if not contract.series_dimension:
                    errors.append("Multi-line chart requires a series_dimension.")
                if not contract.x_axis and not contract.group_by:
                    errors.append("Multi-line chart requires an X-axis dimension.")

            elif chart == "scatter":
                if contract.x_axis and not SchemaResolver.is_numeric_column(contract.x_axis, schema):
                    errors.append(f"Scatter plot X-axis '{contract.x_axis}' must be numeric.")
                if contract.y_axis and not SchemaResolver.is_numeric_column(contract.y_axis, schema):
                    errors.append(f"Scatter plot Y-axis '{contract.y_axis}' must be numeric.")

        return len(errors) == 0, errors

    @classmethod
    def validate_and_repair(
        cls,
        contract: QueryContract,
        schema: dict[str, Any],
    ) -> tuple[QueryContract, list[str]]:
        """
        Validate and deterministically repair/align AnalysisSpec against the active dataset schema.
        Aligns field casing, ensures aggregation-datatype validity, aligns categorical filter values,
        preserves grouping & series dimensions, and enforces chart compatibility.
        """
        repairs: list[str] = []

        if not contract.is_related_to_dataset or contract.intent == "UNRELATED":
            return contract, repairs

        raw_cols = schema.get("columns", []) if isinstance(schema, dict) else []
        schema_col_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in raw_cols if c]
        cols_lower = {c.lower(): c for c in schema_col_names}

        # 1. FIELD REPAIRS & CASING ALIGNMENT
        # Dimensions
        repaired_dims = []
        for dim in contract.dimensions:
            if dim in schema_col_names:
                repaired_dims.append(dim)
            elif dim.lower() in cols_lower:
                canonical = cols_lower[dim.lower()]
                repaired_dims.append(canonical)
                repairs.append(f"Aligned dimension casing '{dim}' -> '{canonical}'.")
            else:
                res = SchemaResolver.resolve_column(dim, schema_col_names, schema)
                if res:
                    repaired_dims.append(res)
                    repairs.append(f"Resolved dimension '{dim}' -> '{res}'.")
        contract.dimensions = repaired_dims

        # Group By
        repaired_gb = []
        for grp in contract.group_by:
            if grp in schema_col_names:
                repaired_gb.append(grp)
            elif grp.lower() in cols_lower:
                canonical = cols_lower[grp.lower()]
                repaired_gb.append(canonical)
                repairs.append(f"Aligned group_by casing '{grp}' -> '{canonical}'.")
            else:
                res = SchemaResolver.resolve_column(grp, schema_col_names, schema)
                if res:
                    repaired_gb.append(res)
                    repairs.append(f"Resolved group_by '{grp}' -> '{res}'.")
        contract.group_by = repaired_gb

        # Series Dimension
        if contract.series_dimension:
            sd = contract.series_dimension
            if sd in schema_col_names:
                pass
            elif sd.lower() in cols_lower:
                contract.series_dimension = cols_lower[sd.lower()]
                repairs.append(f"Aligned series_dimension casing '{sd}' -> '{contract.series_dimension}'.")
            else:
                res = SchemaResolver.resolve_column(sd, schema_col_names, schema)
                if res:
                    contract.series_dimension = res
                    repairs.append(f"Resolved series_dimension '{sd}' -> '{res}'.")
                else:
                    contract.series_dimension = None

        # Measure Column
        if contract.measure_column:
            mc = contract.measure_column.strip()
            if mc not in ("*", "total_records"):
                if mc in schema_col_names:
                    pass
                elif mc.lower() in cols_lower:
                    canonical = cols_lower[mc.lower()]
                    contract.measure_column = canonical
                    contract.measure = canonical
                    contract.measure_expression = canonical
                    repairs.append(f"Aligned measure casing '{mc}' -> '{canonical}'.")
                else:
                    res = SchemaResolver.resolve_column(mc, schema_col_names, schema)
                    if res:
                        contract.measure_column = res
                        contract.measure = res
                        contract.measure_expression = res
                        repairs.append(f"Resolved measure '{mc}' -> '{res}'.")

        # Filters Casing & Value Alignment
        for f in contract.filters:
            f_col = getattr(f, "column", None)
            if f_col:
                if f_col not in schema_col_names and f_col.lower() in cols_lower:
                    canonical = cols_lower[f_col.lower()]
                    f.column = canonical
                    repairs.append(f"Aligned filter column casing '{f_col}' -> '{canonical}'.")
                elif f_col not in schema_col_names:
                    res = SchemaResolver.resolve_column(f_col, schema_col_names, schema)
                    if res:
                        f.column = res
                        repairs.append(f"Resolved filter column '{f_col}' -> '{res}'.")

                # Align categorical filter value casing if known
                if isinstance(f.value, str):
                    for c_info in raw_cols:
                        if isinstance(c_info, dict) and c_info.get("name", "").lower() == f.column.lower():
                            samples = c_info.get("sample_values") or c_info.get("sample") or []
                            for s in samples:
                                if str(s).strip().lower() == f.value.strip().lower():
                                    if str(s) != f.value:
                                        repairs.append(f"Aligned filter value '{f.value}' -> '{s}' on column '{f.column}'.")
                                        f.value = str(s)
                                    break

        # 2. AGGREGATION & DATATYPE VALIDITY
        if contract.aggregation:
            agg_upper = contract.aggregation.upper()
            if agg_upper not in cls.ALLOWED_AGGREGATIONS:
                contract.aggregation = "COUNT"
                repairs.append(f"Invalid aggregation '{agg_upper}' repaired to 'COUNT'.")
                agg_upper = "COUNT"

            if agg_upper in ("SUM", "AVG"):
                if not contract.measure_column or contract.measure_column in ("*", "total_records"):
                    contract.aggregation = "COUNT"
                    contract.measure = "total_records"
                    contract.measure_column = None
                    contract.measure_expression = None
                    repairs.append(f"Deterministic repair: {agg_upper}(*) converted to COUNT(*).")
                else:
                    is_num = SchemaResolver.is_numeric_column(contract.measure_column, schema)
                    if not is_num:
                        contract.aggregation = "COUNT"
                        contract.measure = "total_records"
                        contract.measure_column = None
                        contract.measure_expression = None
                        repairs.append(f"Deterministic repair: Non-numeric measure on {agg_upper} converted to COUNT(*).")

            elif agg_upper in ("MIN", "MAX"):
                if not contract.measure_column or contract.measure_column in ("*", "total_records"):
                    contract.aggregation = "COUNT"
                    contract.measure = "total_records"
                    contract.measure_column = None
                    contract.measure_expression = None
                    repairs.append(f"Deterministic repair: {agg_upper}(*) converted to COUNT(*).")

        # 3. GROUPING & SERIES INTEGRITY
        if contract.group_by:
            contract.answer_type = "GROUPED"
            contract.should_visualize = True
            for g in contract.group_by:
                if g not in contract.dimensions:
                    contract.dimensions.append(g)

        if contract.series_dimension:
            if contract.series_dimension not in contract.group_by:
                contract.group_by.append(contract.series_dimension)
                repairs.append(f"Added series_dimension '{contract.series_dimension}' into group_by.")
            if contract.series_dimension not in contract.dimensions:
                contract.dimensions.append(contract.series_dimension)

        # 4. CHART COMPATIBILITY REPAIR
        if contract.suggested_chart_type:
            chart = contract.suggested_chart_type.lower()
            if chart == "histogram":
                target = contract.x_axis or contract.measure_column or (contract.dimensions[0] if contract.dimensions else None)
                if target:
                    is_num = SchemaResolver.is_numeric_column(target, schema)
                    if not is_num:
                        card = SchemaResolver.get_column_cardinality(target, schema)
                        new_chart = "pie" if (card and card <= 4) else "bar"
                        contract.suggested_chart_type = new_chart
                        contract.intent = "AGGREGATION"
                        contract.answer_type = "GROUPED"
                        contract.aggregation = "COUNT"
                        contract.measure = "total_records"
                        contract.measure_column = None
                        contract.measure_expression = None
                        contract.group_by = [target]
                        contract.dimensions = [target]
                        contract.x_axis = target
                        repairs.append(f"Deterministic repair: Categorical histogram on '{target}' repaired to '{new_chart}'.")

            elif chart == "multi_line":
                if len(contract.group_by) >= 2 and not contract.series_dimension:
                    contract.series_dimension = contract.group_by[1]
                    contract.x_axis = contract.group_by[0]
                    repairs.append(f"Assigned series_dimension '{contract.series_dimension}' from group_by.")

        logger.info("[ANALYSIS SPEC VALIDATOR] Repaired AnalysisSpec: %s repairs applied.", len(repairs))
        for r in repairs:
            logger.info("  - %s", r)

        return contract, repairs
