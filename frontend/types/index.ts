/**
 * TypeScript type definitions for the application.
 */

export interface User {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  file_name: string;
  file_type: string;
  table_name: string;
  row_count: number;
  column_count: number;
  schema_info: SchemaInfo | null;
  created_at: string;
}

export interface SchemaInfo {
  columns: ColumnInfo[];
  cleaning?: {
    original_rows: number;
    clean_rows: number;
    removed_blank_or_footer_rows: number;
    removed_duplicate_rows: number;
    converted_columns: string[];
    source_encoding: string | null;
    cleaned_file_name: string;
  };
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  nullable: boolean;
  unique_count: number;
  sample_values: string[];
}

export interface DatasetProfile {
  id: string;
  dataset_id: string;
  profile_json: Record<string, any>;
  created_at: string;
}

export interface QueryResult {
  query: {
    id: string;
    dataset_id: string;
    question: string;
    intent: string | null;
    created_at: string;
  };
  report: {
    id: string;
    query_id: string;
    answer: string;
    insights: string[] | null;
    chart_paths: string[] | null;
    data_used: Record<string, any> | null;
    agents_used: string[] | null;
    created_at: string;
  } | null;
  agents: AgentRun[];
}

export interface AgentRun {
  agent_name: string;
  status: string;
  execution_time_ms: number | null;
}

export type FullQueryResponse = QueryResult;
